// Sampling pipeline execution. Today this hosts the greedy fast path (the
// fused argmax op) plus the CPU reference path used by tests; heterogeneous
// batches raise UnimplementedError until the fused bias/penalty/filter/sample
// kernel lands in sampling_kernels.cu.
#include "inferx/sampling/sampler.h"

#include <cmath>
#include <cstring>
#include <vector>

#include "inferx/core/device_runtime.h"
#include "inferx/ops/sampling.h"
#include "inferx/sampling/sampler_output.h"

namespace inferx::sampling {
namespace {

float Bf16BitsToFloat(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

}  // namespace

Sampler::Sampler(Tensor values, Tensor indices, Tensor results,
                 std::int64_t vocab_size)
    : values_(std::move(values)),
      indices_(std::move(indices)),
      results_(std::move(results)),
      vocab_size_(vocab_size) {}

StatusOr<std::unique_ptr<Sampler>> Sampler::Create(int max_num_seqs,
                                                   std::int64_t vocab_size,
                                                   DeviceId device) {
  if (max_num_seqs <= 0 || vocab_size <= 0)
    return InvalidArgumentError("sampler requires positive capacities");
  if (!device.IsCpu() && !device.IsCuda())
    return UnimplementedError("unsupported sampling device");
  Tensor values, indices, results;
  if (device.IsCuda()) {
    const std::int64_t parts = (vocab_size + 4095) / 4096;
    INFERX_ASSIGN_OR_RETURN(values,
        Tensor::Empty(DataType::kFloat, Shape({max_num_seqs * parts}), device));
    INFERX_ASSIGN_OR_RETURN(indices,
        Tensor::Empty(DataType::kInt32, Shape({max_num_seqs * parts}), device));
  }
  INFERX_ASSIGN_OR_RETURN(results,
      Tensor::Empty(DataType::kInt32, Shape({max_num_seqs}), device));
  return std::unique_ptr<Sampler>(
      new Sampler(std::move(values), std::move(indices), std::move(results), vocab_size));
}

Status Sampler::Sample(ops::ExecutionContext& ctx, const Tensor& logits,
                       const SamplingMetadata& metadata, SamplerOutput& output) {
  if (!logits.IsDefined() || logits.Rank() != 2 ||
      (logits.GetDataType() != DataType::kBFloat16 &&
       logits.GetDataType() != DataType::kFloat) ||
      logits.Device() != ctx.device()) {
    return InvalidArgumentError("sampler requires a float32/bfloat16 [batch, vocab] matrix");
  }
  const int batch = static_cast<int>(logits.Dim(0));
  if (logits.Dim(1) != vocab_size_ || metadata.batch != batch ||
      static_cast<int>(metadata.requests.size()) != batch) {
    return InvalidArgumentError("logits and sampling metadata disagree on the batch");
  }
  INFERX_ASSIGN_OR_RETURN(output.sampled_token_ids, results_.Slice(0, batch));
  output.logprobs = Tensor{};
  if (batch == 0) return OkStatus();
  if (!metadata.all_greedy)
    return UnimplementedError("random sampling (temperature/top-k/top-p) is not implemented yet");

  if (ctx.device().IsCuda()) {
    return ops::GreedyArgmax(ctx, logits, values_, indices_, output.sampled_token_ids);
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
  int32_t* out = output.sampled_token_ids.DataAs<int32_t>();
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
  return OkStatus();
}

}  // namespace inferx::sampling
