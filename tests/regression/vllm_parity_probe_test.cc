// Regression probe for the InferX-vs-vLLM greedy parity gaps recorded in
// benchmarks/qwen3/serve_results/refactor_fix_20261009 (four of eight
// completions differed). The probe pins down, per differing prompt, that
//
//   1. the decision point (first differing token) sits at a <= 2-ulp BF16
//      near-tie under a single-chunk forward, where each stack's own
//      rounding legitimately decides, and
//   2. the serving execution path itself -- prompt prefill followed by
//      one-token decode steps against a persistent KV pool, exactly what
//      ModelRunner does -- is self-consistent: the recorded InferX choice
//      is the argmax of the logits that path actually produced.
//
// Together with argmax_tie_test (the sampler's lowest-index tie break,
// matching vLLM's torch.argmax semantics), these are the executable
// record of the investigation in docs/bugs.md.
#include <algorithm>
#include <bit>
#include <cmath>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "nlohmann/json.hpp"
#include "inferx/core/device_runtime.h"
#include "inferx/models/load_options.h"
#include "inferx/models/registry.h"
#include "inferx/models/scratch_arena.h"
#include "inferx/ops/flash_attention.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx {
namespace models {
namespace {

using Json = nlohmann::json;

constexpr const char* kResults =
    "benchmarks/qwen3/serve_results/refactor_fix_20261009";
constexpr int64_t kBlockSize = 16;

std::vector<TokenId> EncodeText(Tokenizer& tokenizer, const std::string& text) {
  auto ids = tokenizer.Encode(text);
  if (!ids.ok()) {
    ADD_FAILURE() << ids.status();
    return {};
  }
  return std::vector<TokenId>(ids->begin(), ids->end());
}

float Bf16ToFloat(uint16_t bits) {
  return std::bit_cast<float>(static_cast<uint32_t>(bits) << 16);
}

/// Walks a token sequence the way the runner does: the first Append is the
/// prompt prefill (one chunk), every later Append is a decode step of one
/// token, all against one persistent KV pool whose page table grows.
class SteppingSession {
 public:
  SteppingSession(Model& model, DeviceRuntime& runtime, DeviceStream stream,
                  Allocator& scratch, ops::AttentionPlanWorkspace& plan, int64_t max_pages)
      : model_(model), runtime_(runtime), stream_(stream), scratch_(scratch),
        plan_(plan), max_pages_(max_pages) {
    const ModelSpec& spec = model_.GetModelSpec();
    auto pool = KvBlockPool::Create(spec.state.paged_kv.size(), max_pages, kBlockSize,
                                    spec.state.paged_kv.front(), spec.device);
    if (!pool.ok()) {
      ADD_FAILURE() << pool.status();
      return;
    }
    pool_.emplace(std::move(*pool));
    SingleRankComm comm;
    comm_ = std::move(comm);
  }

  /// Appends `chunk` (prefill on first call, one token after) and returns
  /// the [vocab] logits at the new last position, as BF16 bits.
  std::vector<uint16_t> Append(const std::vector<TokenId>& chunk) {
    const ModelSpec& spec = model_.GetModelSpec();
    const int64_t tokens = static_cast<int64_t>(chunk.size());
    const int64_t kv_len = computed_ + tokens;
    const int64_t pages = (kv_len + kBlockSize - 1) / kBlockSize;
    if (pages > max_pages_) {
      ADD_FAILURE() << "stepping session overflowed its KV budget";
      return {};
    }
    std::vector<int32_t> ids(chunk.begin(), chunk.end()), positions(tokens),
        batch(tokens, 0), qo{0, static_cast<int32_t>(tokens)},
        kv{0, static_cast<int32_t>(pages)}, tables(pages),
        last{(static_cast<int32_t>(kv_len) - 1) % kBlockSize + 1};
    for (int64_t i = 0; i < tokens; ++i) positions[i] = static_cast<int32_t>(computed_ + i);
    for (int64_t i = 0; i < pages; ++i) tables[i] = static_cast<int32_t>(i);

    const auto upload = [&](const std::vector<int32_t>& values) -> StatusOr<Tensor> {
      INFERX_ASSIGN_OR_RETURN(auto tensor,
                              Tensor::Empty(DataType::kInt32,
                                            Shape({static_cast<int64_t>(values.size())}),
                                            spec.device));
      if (!values.empty()) {
        INFERX_RETURN_IF_ERROR(runtime_.CopyAsync(
            tensor.Data(), values.data(), values.size() * sizeof(int32_t),
            CopyKind::kHostToDevice, stream_));
      }
      return tensor;
    };

    StepInputs inputs;
    auto ids_tensor = upload(ids);
    auto positions_tensor = upload(positions);
    AttentionMetadata attn;
    auto batch_tensor = upload(batch);
    auto qo_tensor = upload(qo);
    auto kv_tensor = upload(kv);
    auto tables_tensor = upload(tables);
    auto last_tensor = upload(last);
    if (!ids_tensor.ok() || !positions_tensor.ok() || !batch_tensor.ok() ||
        !qo_tensor.ok() || !kv_tensor.ok() || !tables_tensor.ok() || !last_tensor.ok()) {
      ADD_FAILURE() << "upload failed";
      return {};
    }
    inputs.token_ids = *std::move(ids_tensor);
    inputs.positions = *std::move(positions_tensor);
    attn.batch_indices = *std::move(batch_tensor);
    attn.qo_indptr = *std::move(qo_tensor);
    attn.kv_indptr = *std::move(kv_tensor);
    attn.kv_indices = *std::move(tables_tensor);
    attn.last_page_len = *std::move(last_tensor);
    attn.host_qo_indptr = absl::MakeConstSpan(qo);
    attn.num_tokens = static_cast<int>(tokens);
    attn.num_seqs = 1;

    ops::OpContext op(runtime_, stream_);
    if (auto began = ops::BeginAttentionStep(op, *attn.kv_indptr, *attn.last_page_len,
                                             kBlockSize, attn.num_tokens, 1, plan_);
        !began.ok()) {
      ADD_FAILURE() << began;
      return {};
    }
    ForwardContext ctx(op, attn, *pool_, comm_, scratch_, plan_);
    auto hidden = Tensor::Empty(spec.dtype, Shape({tokens, spec.hidden_size}), &scratch_);
    if (!hidden.ok()) {
      ADD_FAILURE() << hidden.status();
      return {};
    }
    if (auto ran = model_.Forward(ctx, inputs, *hidden); !ran.ok()) {
      ADD_FAILURE() << ran;
      return {};
    }
    auto rows = upload({static_cast<int32_t>(tokens) - 1});
    if (!rows.ok()) {
      ADD_FAILURE() << rows.status();
      return {};
    }
    auto logits = Tensor::Empty(spec.dtype, Shape({1, spec.vocab_size}), &scratch_);
    if (!logits.ok()) {
      ADD_FAILURE() << logits.status();
      return {};
    }
    if (auto got = model_.ComputeLogits(ctx, *hidden, *rows, *logits); !got.ok()) {
      ADD_FAILURE() << got;
      return {};
    }
    std::vector<uint16_t> bits(spec.vocab_size);
    if (auto read = runtime_.Copy(bits.data(), logits->Data(), bits.size() * 2,
                                  CopyKind::kDeviceToHost);
        !read.ok()) {
      ADD_FAILURE() << read;
      return {};
    }
    (void)runtime_.SynchronizeStream(stream_);
    computed_ = kv_len;
    return bits;
  }

 private:
  Model& model_;
  DeviceRuntime& runtime_;
  DeviceStream stream_;
  Allocator& scratch_;
  ops::AttentionPlanWorkspace& plan_;
  int64_t max_pages_ = 0;
  std::optional<KvBlockPool> pool_;
  SingleRankComm comm_;
  int64_t computed_ = 0;
};

class VllmParityProbeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto runtime = RuntimeFor(Device::Cuda(0));
    ASSERT_TRUE(runtime.ok()) << runtime.status();
    runtime_ = *runtime;
    ASSERT_TRUE(runtime_->Activate().ok());
    stream_ = *runtime_->CreateStream();

    auto tokenizer = Tokenizer::FromFile("models/Qwen3-0.6B/tokenizer.json");
    ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
    tokenizer_ = *tokenizer;

    ModelLoadOptions options;
    options.directory = "models/Qwen3-0.6B";
    options.device = Device::Cuda(0);
    auto made = LoadModel(options);
    ASSERT_TRUE(made.ok()) << made.status();
    model_ = std::move(*made);

    scratch_ = std::make_unique<ScratchArena>(ScratchArena::Measuring(options.device));
    auto plan = Tensor::Empty(DataType::kInt32, Shape({3 * 512 + 1}), scratch_.get());
    ASSERT_TRUE(plan.ok()) << plan.status();
    plan_.plan = std::move(*plan);
  }

  DeviceRuntime* runtime_ = nullptr;
  DeviceStream stream_{};
  std::shared_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<Model> model_;
  std::unique_ptr<ScratchArena> scratch_;
  ops::AttentionPlanWorkspace plan_;
};

TEST_F(VllmParityProbeTest, DivergenceDecisionPoints) {
  Json manifest, inferx, vllm;
  {
    std::ifstream input(std::string(kResults) + "/manifest.json");
    input >> manifest;
  }
  {
    std::ifstream input(std::string(kResults) + "/inferx_1/agree-inferx.json");
    input >> inferx;
  }
  {
    std::ifstream input(std::string(kResults) + "/vllm_1/agree-vllm.json");
    input >> vllm;
  }
  ASSERT_EQ(manifest["prompts"].size(), 8u);
  ASSERT_EQ(inferx.size(), 8u);
  ASSERT_EQ(vllm.size(), 8u);

  for (int i = 0; i < 8; ++i) {
    const std::string prompt = manifest["prompts"][i];
    const std::string ix_text = inferx[i];
    const std::string vm_text = vllm[i];
    if (ix_text == vm_text) {
      std::printf("[probe %d] MATCH\n", i);
      continue;
    }

    const std::vector<TokenId> full_ix = EncodeText(*tokenizer_, prompt + ix_text);
    const std::vector<TokenId> full_vm = EncodeText(*tokenizer_, prompt + vm_text);
    const size_t shared =
        std::mismatch(full_ix.begin(), full_ix.end(), full_vm.begin(), full_vm.end()).first -
        full_ix.begin();
    ASSERT_LT(shared + 1, full_ix.size());
    ASSERT_LT(shared + 1, full_vm.size());
    const TokenId ix_token = full_ix[shared];
    const TokenId vm_token = full_vm[shared];
    const std::vector<TokenId> prefix(full_vm.begin(), full_vm.begin() + shared);

    // (1) Single-chunk forward of the shared prefix: the cross-stack
    // view. Every divergence must sit at a <= 2-ulp near-tie, where the
    // two stacks' last-bit roundings each decide.
    SteppingSession chunked(*model_, *runtime_, stream_, *scratch_, plan_, 256);
    const std::vector<uint16_t> bits = chunked.Append(prefix);
    ASSERT_EQ(bits.size(), static_cast<size_t>(model_->GetModelSpec().vocab_size));
    int64_t argmax = 0;
    for (size_t v = 1; v < bits.size(); ++v) {
      if (Bf16ToFloat(bits[v]) > Bf16ToFloat(bits[argmax])) argmax = static_cast<int64_t>(v);
    }
    const int ulps = std::abs(static_cast<int>(bits[ix_token]) -
                              static_cast<int>(bits[vm_token]));
    std::printf("[probe %d] DIFFER at token %zu: inferx=%d vllm=%d "
                "(logits %.4f / %.4f, %d ulps)\n",
                i, shared, ix_token, vm_token, Bf16ToFloat(bits[ix_token]),
                Bf16ToFloat(bits[vm_token]), ulps);
    // The engine's own choice is maximal under this view, modulo the
    // lowest-index tie order the fused argmax and this scan share.
    EXPECT_GE(bits[ix_token], bits[argmax])
        << "single-chunk logits contradict InferX's own choice";
    EXPECT_LE(ulps, 2) << "diverged without a near-tie (" << ulps << " ulps)";

    // (2) Serving path: prompt prefill, then one-token decode steps --
    // the kernels the runner actually ran. The recorded choice must be
    // the argmax of THIS path's logits.
    SteppingSession serving(*model_, *runtime_, stream_, *scratch_, plan_, 256);
    // Prefill the original prompt exactly as the server tokenized it.
    const std::vector<TokenId> encoded_prompt = EncodeText(*tokenizer_, prompt);
    std::vector<uint16_t> step_bits = serving.Append(encoded_prompt);
    ASSERT_EQ(step_bits.size(), static_cast<size_t>(model_->GetModelSpec().vocab_size));
    // Decode the completion tokens up to the decision point.
    for (size_t k = encoded_prompt.size(); k < prefix.size(); ++k) {
      step_bits = serving.Append({prefix[k]});
    }
    int64_t step_argmax = 0;
    for (size_t v = 1; v < step_bits.size(); ++v) {
      if (Bf16ToFloat(step_bits[v]) > Bf16ToFloat(step_bits[step_argmax])) {
        step_argmax = static_cast<int64_t>(v);
      }
    }
    std::printf("[probe %d] serving path: argmax=%ld chosen=%d (%d ulps between them)\n",
                i, static_cast<long>(step_argmax), ix_token,
                std::abs(static_cast<int>(step_bits[ix_token]) -
                         static_cast<int>(step_bits[step_argmax])));
    EXPECT_GE(step_bits[ix_token], step_bits[step_argmax])
        << "serving-path logits contradict the engine's recorded choice";
  }
}

}  // namespace
}  // namespace models
}  // namespace inferx
