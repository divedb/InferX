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
#include "inferx/ops/attention/cache_spec_bridge.h"
#include "inferx/models/scratch_arena.h"
#include "inferx/models/registry.h"
#include "nlohmann/json.hpp"

namespace inferx {
namespace diagnostic {
namespace {

using Json = nlohmann::json;

/// One case's forward against a single-sequence paged layout. The session
/// owns the attention runtime; the facade drives its single request and
/// the physical page order comes from the runtime's own block table.
class ReplaySession {
 public:
  ReplaySession(std::unique_ptr<models::Model> model, DeviceRuntime& runtime, DeviceStream stream,
                int64_t max_len, int64_t page_size)
      : model_(std::move(model)),
        runtime_(runtime),
        stream_(stream),
        page_size_(page_size) {
    comm_ = std::make_unique<SingleRankComm>();
    op_ = std::make_unique<ops::OpContext>(runtime_, stream_);
    // Measured scratch: the session measures while it serves and never
    // freezes (the diagnostic replays no CUDA graphs).
    scratch_ = std::make_unique<models::ScratchArena>(
        models::ScratchArena::Measuring(runtime_.device()));

    auto registry = std::make_unique<attn::BackendRegistry>();
    attn::RegisterBuiltinBackends(*registry);
    attn::AttentionConfig config;
    config.block_size = page_size;
    config.max_model_len = max_len;
    config.max_seqs = 1;
    config.max_batch_tokens = std::max<int64_t>(max_len, 4096);
    config.cache_bytes_budget = 1LL << 32;
    config.device = runtime_.device();
    config.enable_prefix_caching = false;
    config.forced_backend = "flash_dense";
    attention_ = Take(attn::AttentionRuntime::Create(std::move(registry), config,
                                                     model_->GetAttentionSpecs()));
    const std::vector<attn::AttentionSpec> specs = model_->GetAttentionSpecs();
    for (size_t l = 0; l < specs.size(); ++l) {
      ops_.push_back(Take(attention_->MakeOp(static_cast<int64_t>(l))));
    }
  }

  /// \brief Runs tokens [start, end) as one chunk and returns the logits at
  ///        the chunk's last position.
  Tensor Forward(absl::Span<const int32_t> tokens, int start, int count) {
    const int end = start + count;
    std::vector<int32_t> ids(tokens.begin() + start, tokens.begin() + end),
        positions(count);
    std::iota(positions.begin(), positions.end(), start);

    Check(attention_->Cache().Reserve(kRequestId, GrowthTarget{start, end}));
    const std::vector<std::int32_t> query_lens{count};
    const std::vector<std::int32_t> seq_lens{end};
    attn::BatchDesc desc;
    desc.num_reqs = 1;
    desc.num_tokens = count;
    desc.query_lens = query_lens;
    desc.seq_lens = seq_lens;
    desc.max_query_len = count;
    desc.max_seq_len = end;
    desc.block_size = static_cast<std::int32_t>(page_size_);
    desc.num_prefills = count > 1 ? 1 : 0;
    desc.num_prefill_tokens = count;
    const auto blocks = attention_->Cache().GroupBlocks(kRequestId, 0);
    desc.block_ids[0] = blocks;
    desc.block_ids_num_pages[0] = static_cast<std::int32_t>(blocks.size());

    models::StepInputs inputs;
    inputs.token_ids = Upload(ids);
    inputs.positions = Upload(positions);
    attn::StepContext step = Take(attention_->BeginStep(desc, *op_));

    models::ForwardContext ctx(*op_, step, *comm_, *scratch_,
                                absl::MakeConstSpan(ops_));
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

  const attn::AttentionRuntime& Attention() const { return *attention_; }
  const KvBlockPool& Pool() const { return *attention_->BlockPool(); }
  /// \brief The case's logical page table, in order.
  std::vector<int32_t> PageTable() const {
    const auto blocks = attention_->Cache().GroupBlocks(kRequestId, 0);
    return std::vector<int32_t>(blocks.begin(), blocks.end());
  }
  int64_t page_size() const { return page_size_; }
  /// \brief Releases the request between cases: each case starts fresh.
  void Reset() { Check(attention_->Cache().Release(kRequestId)); }

 private:
  Tensor Upload(const std::vector<int32_t>& values) {
    Tensor tensor = Take(Tensor::Empty(
        DataType::kInt32, Shape({static_cast<int64_t>(values.size())}), runtime_.device()));
    if (!values.empty()) {
      Check(runtime_.Memory().CopyAsync(tensor.Data(), values.data(), values.size() * sizeof(int32_t),
                               CopyKind::kHostToDevice, stream_));
    }
    return tensor;
  }

  static constexpr uint64_t kRequestId = 1;

  std::unique_ptr<models::Model> model_;
  DeviceRuntime& runtime_;
  DeviceStream stream_;
  int64_t page_size_ = 16;
  std::unique_ptr<attn::AttentionRuntime> attention_;
  std::vector<attn::AttentionOp> ops_;
  std::unique_ptr<SingleRankComm> comm_;
  std::unique_ptr<ops::OpContext> op_;
  std::unique_ptr<models::ScratchArena> scratch_;
};

void Replay(const ReplayLogitsParams& params) {
  if (params.chunk_size <= 0) throw std::runtime_error("positive chunk size required");
  Json fixture;
  std::ifstream(params.fixture_path) >> fixture;
  if (fixture.at("cases").empty()) throw std::runtime_error("fixture has no cases");
  const Device device = Device::Cuda(0);
  auto* runtime = Take(RuntimeFor(device));
  Check(runtime->Activate());
  const OwnedStream owned_stream = Take(runtime->Streams().CreateStream(StreamPriority::kNormal));
  const DeviceStream stream = owned_stream.Get();

  models::ModelLoadOptions options;
  options.directory = params.model_dir;
  options.device = Device::Cuda(0);
  auto model = Take(models::LoadModel(options));
  const models::ModelSpec& info = model->GetModelSpec();
  const std::vector<attn::AttentionSpec> specs = model->GetAttentionSpecs();
  // bytes/token = 2 (K and V) * kv_heads * head_dim * elem; the K or V
  // plane width in bf16 elements is half that in elements.
  const size_t plane_width =
      static_cast<size_t>(Take(attn::CacheBytesPerToken(specs.front())) / 4);

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
  // Page placement is the pool's, not the diagnostic's: the --page-order
  // knob shaped a hand-built table the facade now owns. Physical order
  // still varies through the LIFO free queue across cases.
  const int page_size = 16;
  ReplaySession session(std::move(model), *runtime, stream, max_length, page_size);

  std::ofstream file(params.output, std::ios::binary);
  file.exceptions(std::ios::badbit | std::ios::failbit);
  int case_index = 0;
  for (const auto& entry : fixture.at("cases")) {
    const auto tokens = entry.at("tokens").get<std::vector<int32_t>>();
    const int prompt_length = entry.at("prompt_length");
    std::optional<Tensor> logits;
    session.Reset();
    for (int start = 0; start < static_cast<int>(tokens.size());) {
      const int count = start < prompt_length
                            ? std::min(params.chunk_size, prompt_length - start)
                            : 1;
      logits = session.Forward(tokens, start, count);
      // Keep all step allocations alive until the forward has finished.
      Check(runtime->Streams().SynchronizeStream(stream));
      start += count;
    }
    if (logits->GetDataType() != DataType::kBFloat16 ||
        logits->Numel() != info.vocab_size) {
      throw std::runtime_error("unexpected replay logit dtype or shape");
    }
    std::vector<uint16_t> bits(info.vocab_size);
    Check(runtime->Memory().Copy(bits.data(), logits->Data(), bits.size() * sizeof(uint16_t),
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
      const std::vector<int32_t> table = session.PageTable();
      const int num_layers = static_cast<int>(specs.size());
      for (int layer = 0; layer < num_layers; ++layer) {
        for (bool key : {true, false}) {
          Tensor cache = Take(session.Pool().LayerCache(layer));
          std::vector<uint16_t> raw(cache.Numel());
          Check(runtime->Memory().Copy(raw.data(), cache.Data(), cache.NBytes(),
                              CopyKind::kDeviceToHost));
          std::ofstream dump(dir / ("layer_" + std::to_string(layer) +
                                    (key ? ".k.bf16" : ".v.bf16")),
                             std::ios::binary);
          dump.exceptions(std::ios::badbit | std::ios::failbit);
          // One page is [K plane, V plane]; each plane holds page_size
          // rows of [kv_heads * head_dim] bf16 elements.
          const size_t width = plane_width;
          for (size_t pos = 0; pos < tokens.size(); ++pos) {
            const size_t page = static_cast<size_t>(table[pos / page_size]);
            const size_t slot = pos % page_size;
            const size_t offset =
                page * (2 * width) + slot * width + (key ? 0 : width);
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
