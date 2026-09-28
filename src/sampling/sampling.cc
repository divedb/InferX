// Sampling pipeline execution. Today this hosts the greedy fast path (the
// fused argmax op) plus the CPU reference path used by tests; heterogeneous
// batches raise UnimplementedError until the fused bias/penalty/filter/sample
// kernel lands in sampling_kernels.cu.
#include "inferx/sampling/sampler.h"

#include <cmath>
#include <cstring>
#include <optional>
#include <utility>
#include <vector>

#include "inferx/core/device_runtime.h"
#include "inferx/ops/sampling.h"
#include "inferx/sampling/sampler_output.h"

namespace {
// Host staging kept alive until the stream drains; the runner synchronizes
// every step before metadata vectors are rebuilt.
struct UploadStaging {
  std::vector<float> temperature, top_p, min_p, penalties;
  std::vector<int32_t> top_k, greedy;
  std::vector<uint64_t> seeds, rng_offsets;
  std::vector<int32_t> bias_ptr, allow_ptr, hist_ptr;
  std::vector<uint64_t> bias_entries, hist_entries;
  std::vector<int32_t> allow_entries;
};
}  // namespace

namespace inferx::sampling {
namespace {

float Bf16BitsToFloat(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace

Sampler::Sampler(std::optional<Tensor> values, std::optional<Tensor> indices, Tensor results,
                 std::int64_t vocab_size, std::optional<cuda::DeviceParams> params,
                 std::optional<Tensor> probs, int64_t max_num_seqs)
    : values_(std::move(values)),
      indices_(std::move(indices)),
      results_(std::move(results)),
      vocab_size_(vocab_size),
      params_(std::move(params)),
      probs_(std::move(probs)),
      max_num_seqs_(max_num_seqs) {}

StatusOr<std::unique_ptr<Sampler>> Sampler::Create(int max_num_seqs,
                                                   std::int64_t vocab_size,
                                                   DeviceId device) {
  if (max_num_seqs <= 0 || vocab_size <= 0)
    return InvalidArgumentError("sampler requires positive capacities");
  if (!device.IsCpu() && !device.IsCuda())
    return UnimplementedError("unsupported sampling device");
  std::optional<Tensor> values, indices;
  if (device.IsCuda()) {
    const std::int64_t parts = (vocab_size + 4095) / 4096;
    INFERX_ASSIGN_OR_RETURN(values,
        Tensor::Empty(DataType::kFloat32, Shape({max_num_seqs * parts}), device));
    INFERX_ASSIGN_OR_RETURN(indices,
        Tensor::Empty(DataType::kInt32, Shape({max_num_seqs * parts}), device));
  }
  INFERX_ASSIGN_OR_RETURN(auto results,
      Tensor::Empty(DataType::kInt32, Shape({max_num_seqs}), device));
  std::optional<cuda::DeviceParams> params;
  std::optional<Tensor> probs;
  if (device.IsCuda()) {
    cuda::DeviceParams p;
    INFERX_ASSIGN_OR_RETURN(p.temperature, Tensor::Empty(DataType::kFloat32, Shape({max_num_seqs}), device));
    INFERX_ASSIGN_OR_RETURN(p.top_k, Tensor::Empty(DataType::kInt32, Shape({max_num_seqs}), device));
    INFERX_ASSIGN_OR_RETURN(p.top_p, Tensor::Empty(DataType::kFloat32, Shape({max_num_seqs}), device));
    INFERX_ASSIGN_OR_RETURN(p.min_p, Tensor::Empty(DataType::kFloat32, Shape({max_num_seqs}), device));
    INFERX_ASSIGN_OR_RETURN(p.penalties, Tensor::Empty(DataType::kFloat32, Shape({max_num_seqs * 3}), device));
    INFERX_ASSIGN_OR_RETURN(p.seeds, Tensor::Empty(DataType::kUInt64, Shape({max_num_seqs}), device));
    INFERX_ASSIGN_OR_RETURN(p.rng_offsets, Tensor::Empty(DataType::kUInt64, Shape({max_num_seqs}), device));
    INFERX_ASSIGN_OR_RETURN(p.greedy, Tensor::Empty(DataType::kInt32, Shape({max_num_seqs}), device));
    INFERX_ASSIGN_OR_RETURN(p.bias_ptr, Tensor::Empty(DataType::kInt32, Shape({max_num_seqs + 1}), device));
    INFERX_ASSIGN_OR_RETURN(p.allow_ptr, Tensor::Empty(DataType::kInt32, Shape({max_num_seqs + 1}), device));
    INFERX_ASSIGN_OR_RETURN(p.hist_ptr, Tensor::Empty(DataType::kInt32, Shape({max_num_seqs + 1}), device));
    // CSR entry capacities: bias/allow grow with request knobs; the history
    // can reach the context length per request in the worst case.
    const int64_t bias_cap = int64_t(max_num_seqs) * vocab_size;
    INFERX_ASSIGN_OR_RETURN(p.bias_entries, Tensor::Empty(DataType::kUInt64, Shape({bias_cap}), device));
    INFERX_ASSIGN_OR_RETURN(p.allow_entries, Tensor::Empty(DataType::kInt32, Shape({bias_cap}), device));
    INFERX_ASSIGN_OR_RETURN(p.hist_entries, Tensor::Empty(DataType::kUInt64, Shape({bias_cap}), device));
    INFERX_ASSIGN_OR_RETURN(probs, Tensor::Empty(DataType::kFloat32, Shape({int64_t(max_num_seqs), vocab_size}), device));
    params = std::move(p);
  }
  return std::unique_ptr<Sampler>(new Sampler(std::move(values), std::move(indices),
                                              std::move(results), vocab_size, std::move(params),
                                              std::move(probs), max_num_seqs));
}

StatusOr<SamplerOutput> Sampler::Sample(ops::ExecutionContext& ctx, const Tensor& logits,
                                        const SamplingMetadata& metadata) {
  if (logits.Rank() != 2 ||
      (logits.GetDataType() != DataType::kBFloat16 &&
       logits.GetDataType() != DataType::kFloat32) ||
      logits.Device() != ctx.device()) {
    return InvalidArgumentError("sampler requires a float32/bfloat16 [batch, vocab] matrix");
  }
  const int batch = static_cast<int>(logits.Dim(0));
  if (logits.Dim(1) != vocab_size_ || metadata.batch != batch ||
      static_cast<int>(metadata.requests.size()) != batch) {
    return InvalidArgumentError("logits and sampling metadata disagree on the batch");
  }
  INFERX_ASSIGN_OR_RETURN(Tensor sampled, results_.Slice(0, batch));
  if (batch == 0) return SamplerOutput{std::move(sampled), std::nullopt};
  if (!metadata.all_greedy) {
    if (ctx.device().IsCuda()) {
      INFERX_RETURN_IF_ERROR(UploadParams(ctx, metadata));
      return cuda::SampleRows(ctx, logits, *params_, *probs_, sampled) -
             Status();  // wrap keeps SamplerOutput assembly below.
    }
    return SampleCpuReference(logits, metadata, sampled);
  }

  if (ctx.device().IsCuda()) {
    INFERX_RETURN_IF_ERROR(
        ops::GreedyArgmax(ctx, logits, *values_, *indices_, sampled));
    return SamplerOutput{std::move(sampled), std::nullopt};
  }

  // CPU reference path: retains the GPU op's lowest-index tie break and its
  // NaN-in-column-zero rule, so CPU and CUDA agree token-for-token.
  INFERX_RETURN_IF_ERROR(ctx.runtime().SynchronizeStream(ctx.stream()));
  const std::int64_t vocab = logits.Dim(1);
  const bool is_bf16 = logits.GetDataType() == DataType::kBFloat16;
  const size_t elem = is_bf16 ? sizeof(uint16_t) : sizeof(float);
  std::vector<std::byte> host(static_cast<size_t>(logits.NBytes()));
  INFERX_RETURN_IF_ERROR(ctx.runtime().Copy(host.data(), logits.Data(), host.size(),
                                            CopyKind::kDeviceToHost));
  auto value_at = [&](int64_t row, int64_t col) -> float {
    const std::byte* p = host.data() + (row * vocab + col) * elem;
    if (!is_bf16) {
      float f = 0.0f;
      std::memcpy(&f, p, sizeof(f));
      return f;
    }
    uint16_t h = 0;
    std::memcpy(&h, p, sizeof(h));
    return Bf16BitsToFloat(h);
  };
  int32_t* out = sampled.DataAs<int32_t>();
  for (int i = 0; i < batch; ++i) {
    float best = value_at(i, 0);
    int32_t best_idx = 0;
    for (int64_t j = 1; j < vocab; ++j) {
      const float value = value_at(i, j);
      if (value > best) {
        best = value;
        best_idx = static_cast<int32_t>(j);
      }
    }
    out[i] = std::isnan(value_at(i, 0)) ? 0 : best_idx;
  }
  return SamplerOutput{std::move(sampled), std::nullopt};
}

}  // namespace inferx::sampling
