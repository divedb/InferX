// Teacher-forced numerical diagnostic; the CLI-independent body of
// `inferx diagnostic replay-logits`. Never use this path for throughput
// timing. Drives the model layer directly: one sequence, chunked prefill,
// one decode-style step per continuation token, with a manually permuted
// page table so cache-layout bugs surface as logit drift.
#include <algorithm>
#include <bit>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "inferx/core/device_runtime.h"
#include "inferx/core/status_util.h"
#include "inferx/diagnostic/replay_logits.h"
#include "inferx/dist/comm.h"
#include "inferx/models/model.h"
#include "inferx/models/scratch_arena.h"
#include "inferx/models/registry.h"
#include "nlohmann/json.hpp"

namespace inferx {
namespace diagnostic {
namespace {

using Json = nlohmann::json;

/// One case's forward against a single-sequence paged layout.
class ReplaySession {
 public:
  ReplaySession(std::unique_ptr<models::Model> model, DeviceRuntime& runtime, DeviceStream stream,
                int page_size, int max_pages, std::vector<int32_t> pages)
      : model_(std::move(model)),
        runtime_(runtime),
        stream_(stream),
        page_size_(page_size),
        pages_(std::move(pages)) {
    const models::ModelSpec& spec = model_->GetModelSpec();
    pool_.emplace(Take(KvBlockPool::Create(spec.state.paged_kv.size(), max_pages,
                                          page_size, spec.state.paged_kv.front(),
                                          runtime_.device())));
    comm_ = std::make_unique<SingleRankComm>();
    op_ = std::make_unique<ops::OpContext>(runtime_, stream_);
    // Measured scratch: the session measures while it serves and never
    // freezes (the diagnostic replays no CUDA graphs).
    scratch_ = std::make_unique<models::ScratchArena>(
        models::ScratchArena::Measuring(runtime_.device()));
    const int64_t max_chunk = 4096;
    plan_.plan = Take(Tensor::Empty(DataType::kInt32, Shape({3 * max_chunk + 1}),
                                     scratch_.get()));
  }

  /// \brief Runs tokens [start, end) as one chunk and returns the logits at
  ///        the chunk's last position.
  Tensor Forward(absl::Span<const int32_t> tokens, int start, int count) {
    const int end = start + count;
    std::vector<int32_t> ids(tokens.begin() + start, tokens.begin() + end), positions(count),
        batch(count, 0), qo{0, count};
    std::iota(positions.begin(), positions.end(), start);
    const int used_pages = (end + page_size_ - 1) / page_size_;
    std::vector<int32_t> kv{0, used_pages}, pages(pages_.begin(),
                                                   pages_.begin() + used_pages);
    std::vector<int32_t> last_page{(end - 1) % page_size_ + 1};

    models::StepInputs inputs;
    inputs.token_ids = Upload(ids);
    inputs.positions = Upload(positions);
    models::AttentionMetadata attn;
    attn.batch_indices = Upload(batch);
    attn.qo_indptr = Upload(qo);
    attn.kv_indptr = Upload(kv);
    attn.kv_indices = Upload(pages);
    attn.last_page_len = Upload(last_page);
    host_qo_ = qo;
    attn.host_qo_indptr = absl::MakeConstSpan(host_qo_);
    attn.num_tokens = count;
    attn.num_seqs = 1;

    Check(ops::BeginAttentionStep(*op_, *attn.kv_indptr, *attn.last_page_len, page_size_,
                                  count, 1, plan_));
    models::ForwardContext ctx(*op_, attn, *pool_, *comm_, *scratch_, plan_);
    const models::ModelSpec& spec = model_->GetModelSpec();
    Tensor hidden = Take(Tensor::Empty(spec.dtype, Shape({count, spec.hidden_size}),
                                       scratch_.get()));
    Check(model_->Forward(ctx, inputs, hidden));
    Tensor logits = Take(Tensor::Empty(spec.dtype, Shape({1, spec.vocab_size}),
                                       scratch_.get()));
    const Tensor sample_rows = Upload({count - 1});
    Check(model_->ComputeLogits(ctx, hidden, sample_rows, logits));
    return logits;
  }

  const KvBlockPool& Pool() const { return *pool_; }
  const std::vector<int32_t>& pages() const { return pages_; }
  int page_size() const { return page_size_; }

 private:
  Tensor Upload(const std::vector<int32_t>& values) {
    Tensor tensor = Take(Tensor::Empty(
        DataType::kInt32, Shape({static_cast<int64_t>(values.size())}), runtime_.device()));
    if (!values.empty()) {
      Check(runtime_.CopyAsync(tensor.Data(), values.data(), values.size() * sizeof(int32_t),
                               CopyKind::kHostToDevice, stream_));
    }
    return tensor;
  }

  std::unique_ptr<models::Model> model_;
  DeviceRuntime& runtime_;
  DeviceStream stream_;
  int page_size_ = 16;
  std::vector<int32_t> pages_;
  std::optional<KvBlockPool> pool_;
  std::unique_ptr<SingleRankComm> comm_;
  std::unique_ptr<ops::OpContext> op_;
  std::unique_ptr<models::ScratchArena> scratch_;
  ops::AttentionPlanWorkspace plan_;
  std::vector<int32_t> host_qo_;  ///< Borrowed by attention metadata spans.
};

void Replay(const ReplayLogitsParams& params) {
  if (params.chunk_size <= 0) throw std::runtime_error("positive chunk size required");
  Json fixture;
  std::ifstream(params.fixture_path) >> fixture;
  if (fixture.at("cases").empty()) throw std::runtime_error("fixture has no cases");
  const Device device = Device::Cuda(0);
  auto* runtime = Take(RuntimeFor(device));
  Check(runtime->Activate());
  const DeviceStream stream = Take(runtime->CreateStream());
  struct StreamGuard {
    DeviceRuntime* runtime;
    DeviceStream stream;
    ~StreamGuard() { (void)runtime->DestroyStream(stream); }
  } guard{runtime, stream};

  models::ModelLoadOptions options;
  options.directory = params.model_dir;
  options.device = Device::Cuda(0);
  auto model = Take(models::LoadModel(options));
  const models::ModelSpec& info = model->GetModelSpec();
  if (info.state.paged_kv.empty()) {
    throw std::runtime_error("replay requires a paged-attention model");
  }

  int max_length = 0;
  for (const auto& entry : fixture.at("cases")) {
    const auto tokens = entry.at("tokens").get<std::vector<int32_t>>();
    const int prompt_length = entry.at("prompt_length");
    if (tokens.empty() ||
        tokens.size() > static_cast<size_t>(info.max_position_embeddings) ||
        prompt_length <= 0 || static_cast<size_t>(prompt_length) > tokens.size()) {
      throw std::runtime_error("invalid replay prefix length");
    }
    for (int token : tokens) {
      if (token < 0 || token >= info.vocab_size) throw std::runtime_error("invalid token ID");
    }
    max_length = std::max(max_length, static_cast<int>(tokens.size()));
  }
  const int page_size = 16;
  const int blocks = (max_length + page_size - 1) / page_size;
  std::vector<int32_t> physical_pages(blocks);
  std::iota(physical_pages.begin(), physical_pages.end(), 0);
  if (params.page_order == "reverse") {
    std::reverse(physical_pages.begin(), physical_pages.end());
  } else if (params.page_order == "shuffle") {
    std::mt19937 rng(20260922);
    std::shuffle(physical_pages.begin(), physical_pages.end(), rng);
  }

  ReplaySession session(std::move(model), *runtime, stream, page_size, blocks,
                        physical_pages);

  std::ofstream file(params.output, std::ios::binary);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  int case_index = 0;
  for (const auto& entry : fixture.at("cases")) {
    const auto tokens = entry.at("tokens").get<std::vector<int32_t>>();
    const int prompt_length = entry.at("prompt_length");
    std::optional<Tensor> logits;
    for (int start = 0; start < static_cast<int>(tokens.size());) {
      const int count = start < prompt_length
                            ? std::min(params.chunk_size, prompt_length - start)
                            : 1;
      logits = session.Forward(tokens, start, count);
      // Keep all step allocations alive until the forward has finished.
      Check(runtime->SynchronizeStream(stream));
      start += count;
    }
    if (logits->GetDataType() != DataType::kBFloat16 ||
        logits->Numel() != info.vocab_size) {
      throw std::runtime_error("unexpected replay logit dtype or shape");
    }
    std::vector<uint16_t> bits(info.vocab_size);
    Check(runtime->Copy(bits.data(), logits->Data(), bits.size() * sizeof(uint16_t),
                        CopyKind::kDeviceToHost));
    std::vector<float> values(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) {
      values[i] = std::bit_cast<float>(static_cast<uint32_t>(bits[i]) << 16);
    }
    file.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(float));
    if (!params.cache_dir.empty()) {
      const auto dir = std::filesystem::path(params.cache_dir) / std::to_string(case_index);
      if (!std::filesystem::create_directories(dir)) {
        throw std::runtime_error("cache dump directory exists");
      }
      const int num_layers = static_cast<int>(info.state.paged_kv.size());
      const KvLayout& layout = info.state.paged_kv.front();
      for (int layer = 0; layer < num_layers; ++layer) {
        for (bool key : {true, false}) {
          Tensor cache = Take(key ? session.Pool().KeyCache(layer)
                                  : session.Pool().ValueCache(layer));
          std::vector<uint16_t> raw(cache.Numel());
          Check(runtime->Copy(raw.data(), cache.Data(), cache.NBytes(),
                              CopyKind::kDeviceToHost));
          std::ofstream dump(dir / ("layer_" + std::to_string(layer) +
                                    (key ? ".k.bf16" : ".v.bf16")),
                             std::ios::binary);
          dump.exceptions(std::ios::badbit | std::ios::failbit);
          const size_t width = layout.kv_heads * layout.head_dim;
          for (size_t pos = 0; pos < tokens.size(); ++pos) {
            const size_t offset =
                (physical_pages[pos / page_size] * page_size + pos % page_size) * width;
            dump.write(reinterpret_cast<const char*>(raw.data() + offset), width * 2);
          }
        }
      }
    }
    ++case_index;
  }
  file.close();
}

}  // namespace

Status RunReplayLogits(const ReplayLogitsParams& params) {
  return Guarded([&] { Replay(params); });
}

}  // namespace diagnostic
}  // namespace inferx
