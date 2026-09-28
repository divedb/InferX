// Sampling pipeline execution. Greedy batches take the fused argmax fast
// path; heterogeneous batches run the pipeline kernels (bias/allowlist,
// penalties, temperature, top-k/top-p/min-p, counter-based draws). The CPU
// reference path mirrors both for tests.
#include "inferx/sampling/sampler.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <cstring>
#include <map>
#include <optional>
#include <utility>
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

uint64_t NextUnitHost(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ull;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z = z ^ (z >> 31);
  return z;
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
    const auto f32 = [&](int64_t n) {
      return Tensor::Empty(DataType::kFloat32, Shape({n}), device);
    };
    const auto i32 = [&](int64_t n) {
      return Tensor::Empty(DataType::kInt32, Shape({n}), device);
    };
    const auto u64 = [&](int64_t n) {
      return Tensor::Empty(DataType::kUInt64, Shape({n}), device);
    };
    INFERX_ASSIGN_OR_RETURN(p.temperature, f32(max_num_seqs));
    INFERX_ASSIGN_OR_RETURN(p.top_p, f32(max_num_seqs));
    INFERX_ASSIGN_OR_RETURN(p.min_p, f32(max_num_seqs));
    INFERX_ASSIGN_OR_RETURN(p.penalties, f32(max_num_seqs * 3));
    INFERX_ASSIGN_OR_RETURN(p.top_k, i32(max_num_seqs));
    INFERX_ASSIGN_OR_RETURN(p.greedy, i32(max_num_seqs));
    INFERX_ASSIGN_OR_RETURN(p.bias_ptr, i32(max_num_seqs + 1));
    INFERX_ASSIGN_OR_RETURN(p.allow_ptr, i32(max_num_seqs + 1));
    INFERX_ASSIGN_OR_RETURN(p.hist_ptr, i32(max_num_seqs + 1));
    // CSR capacities: knobs are bounded by the vocabulary per request; the
    // penalty history can reach the context length in the worst case.
    const int64_t cap = int64_t(max_num_seqs) * vocab_size;
    INFERX_ASSIGN_OR_RETURN(p.seeds, u64(max_num_seqs));
    INFERX_ASSIGN_OR_RETURN(p.rng_offsets, u64(max_num_seqs));
    INFERX_ASSIGN_OR_RETURN(p.bias_entries, u64(cap));
    INFERX_ASSIGN_OR_RETURN(p.hist_entries, u64(cap));
    INFERX_ASSIGN_OR_RETURN(p.allow_entries, i32(cap));
    INFERX_ASSIGN_OR_RETURN(probs,
        Tensor::Empty(DataType::kFloat32, Shape({int64_t(max_num_seqs), vocab_size}), device));
    params = std::move(p);
  }
  return std::unique_ptr<Sampler>(new Sampler(std::move(values), std::move(indices),
                                              std::move(results), vocab_size, std::move(params),
                                              std::move(probs), max_num_seqs));
}

Status Sampler::UploadParams(ops::ExecutionContext& ctx, const SamplingMetadata& metadata) {
  const auto& d = metadata.device;
  const int batch = metadata.batch;
  if (batch > max_num_seqs_) {
    return InvalidArgumentError("sampling batch exceeds workspace capacity");
  }
  if (int64_t(d.bias_entries.size()) > params_->bias_entries->Numel() ||
      int64_t(d.allow_entries.size()) > params_->allow_entries->Numel() ||
      int64_t(d.hist_entries.size()) > params_->hist_entries->Numel()) {
    return ResourceExhaustedError("sampling CSR payload exceeds capacity");
  }
  auto upload = [&](const void* src, Tensor& dst, int64_t bytes) {
    return ctx.runtime().CopyAsync(dst.Data(), src, bytes, CopyKind::kHostToDevice,
                                   ctx.stream());
  };
  INFERX_RETURN_IF_ERROR(upload(d.temperature.data(), *params_->temperature, batch * 4));
  INFERX_RETURN_IF_ERROR(upload(d.top_p.data(), *params_->top_p, batch * 4));
  INFERX_RETURN_IF_ERROR(upload(d.min_p.data(), *params_->min_p, batch * 4));
  INFERX_RETURN_IF_ERROR(upload(d.penalties.data(), *params_->penalties, batch * 12));
  INFERX_RETURN_IF_ERROR(upload(d.top_k.data(), *params_->top_k, batch * 4));
  INFERX_RETURN_IF_ERROR(upload(d.greedy.data(), *params_->greedy, batch * 4));
  INFERX_RETURN_IF_ERROR(upload(d.seeds.data(), *params_->seeds, batch * 8));
  INFERX_RETURN_IF_ERROR(upload(d.rng_offsets.data(), *params_->rng_offsets, batch * 8));
  INFERX_RETURN_IF_ERROR(upload(d.bias_ptr.data(), *params_->bias_ptr, (batch + 1) * 4));
  INFERX_RETURN_IF_ERROR(upload(d.allow_ptr.data(), *params_->allow_ptr, (batch + 1) * 4));
  INFERX_RETURN_IF_ERROR(upload(d.hist_ptr.data(), *params_->hist_ptr, (batch + 1) * 4));
  if (!d.bias_entries.empty()) {
    INFERX_RETURN_IF_ERROR(
        upload(d.bias_entries.data(), *params_->bias_entries, d.bias_entries.size() * 8));
  }
  if (!d.allow_entries.empty()) {
    INFERX_RETURN_IF_ERROR(
        upload(d.allow_entries.data(), *params_->allow_entries, d.allow_entries.size() * 4));
  }
  if (!d.hist_entries.empty()) {
    INFERX_RETURN_IF_ERROR(
        upload(d.hist_entries.data(), *params_->hist_entries, d.hist_entries.size() * 8));
  }
  return OkStatus();
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
      INFERX_RETURN_IF_ERROR(cuda::SampleRows(ctx, logits, *params_, *probs_, sampled));
      return SamplerOutput{std::move(sampled), std::nullopt};
    }
    return SampleCpuReference(ctx, logits, metadata, std::move(sampled));
  }

  if (ctx.device().IsCuda()) {
    INFERX_RETURN_IF_ERROR(
        ops::GreedyArgmax(ctx, logits, *values_, *indices_, sampled));
    return SamplerOutput{std::move(sampled), std::nullopt};
  }
  return SampleCpuReference(ctx, logits, metadata, std::move(sampled));
}

// CPU reference: mirrors the CUDA pipeline stage for stage, including the
// greedy op's lowest-index tie break, so CPU and CUDA agree token-for-token.
StatusOr<SamplerOutput> Sampler::SampleCpuReference(ops::ExecutionContext& ctx,
                                                    const Tensor& logits,
                                                    const SamplingMetadata& metadata,
                                                    Tensor sampled) {
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
  for (int i = 0; i < metadata.batch; ++i) {
    const RequestSamplingParams& r = metadata.requests[i];
    const auto& d = metadata.device;
    std::vector<float> p(vocab);
    for (int64_t j = 0; j < vocab; ++j) p[j] = value_at(i, j);
    for (int e = d.bias_ptr[i]; e < d.bias_ptr[i + 1]; ++e) {
      const uint64_t packed = d.bias_entries[e];
      const uint32_t bits = static_cast<uint32_t>(packed & 0xFFFFFFFFull);
      float bias = 0;
      std::memcpy(&bias, &bits, sizeof(bias));
      p[packed >> 32] += bias;
    }
    if (d.allow_ptr[i + 1] > d.allow_ptr[i]) {
      std::vector<char> ok(vocab, 0);
      for (int e = d.allow_ptr[i]; e < d.allow_ptr[i + 1]; ++e) ok[d.allow_entries[e]] = 1;
      for (int64_t j = 0; j < vocab; ++j) {
        if (!ok[j]) p[j] = -INFINITY;
      }
    }
    for (int e = d.hist_ptr[i]; e < d.hist_ptr[i + 1]; ++e) {
      const uint64_t packed = d.hist_entries[e];
      const int token = static_cast<int>(packed >> 32);
      const int count = static_cast<int>(packed & 0xFFFFFFFFull);
      float v = p[token];
      if (r.repetition_penalty != 1.0f) {
        v = v > 0 ? v / r.repetition_penalty : v * r.repetition_penalty;
      }
      v -= r.presence_penalty + r.frequency_penalty * count;
      p[token] = v;
    }
    float max_value = -INFINITY;
    for (float v : p) max_value = std::max(max_value, v);
    double total = 0;
    for (float& v : p) {
      v = v == -INFINITY ? 0.0f : std::exp(v - max_value);
      total += v;
    }
    if (r.greedy) {
      int32_t best = 0;
      for (int64_t j = 1; j < vocab; ++j) {
        if (p[j] > p[best]) best = static_cast<int32_t>(j);
      }
      out[i] = std::isnan(value_at(i, 0)) ? 0 : (total == 0 ? 0 : best);
      continue;
    }
    if (std::getenv("INFERX_SAMPLE_TRACE") != nullptr) {
      std::string line = "softmax row " + std::to_string(i) + ":";
      for (float v : p) line += " " + std::to_string(v);
      std::fprintf(stderr, "%s\n", line.c_str());
    }
    if (r.temperature != 1.0f && r.temperature > 0) {
      const double inv = 1.0 / r.temperature;
      total = 0;
      for (float& v : p) {
        v = v > 0 ? std::pow(v, inv) : 0.0f;
        total += v;
      }
    }
    if (r.min_p > 0) {
      float top = 0;
      for (float v : p) top = std::max(top, v);
      const float threshold = r.min_p * top;
      for (float& v : p) {
        if (v < threshold) v = 0;
      }
    }
    if ((r.top_k > 0 && r.top_k < vocab) || r.top_p < 1.0f) {
      double mass = 0;
      for (float v : p) mass += v;
      double cumulative = 0;
      const int rounds = r.top_k > 0 ? r.top_k : vocab;
      for (int round = 0; round < rounds; ++round) {
        int best = -1;
        for (int64_t j = 0; j < vocab; ++j) {
          if (p[j] > 0 && (best < 0 || p[j] > p[best])) best = static_cast<int>(j);
        }
        if (best < 0) break;
        cumulative += p[best];
        p[best] = -p[best];  // Selected: negate.
        if (r.top_p < 1.0f && cumulative >= r.top_p * mass) break;
      }
      for (float& v : p) v = v < 0 ? -v : 0;
    }
    double norm = 0;
    for (float v : p) norm += v;
    if (norm == 0) {
      out[i] = 0;
      continue;
    }
    uint64_t state = d.seeds[i] ^ (d.rng_offsets[i] * 0x9E3779B97F4A7C15ull);
    const double u = static_cast<double>(NextUnitHost(state) >> 40) / 16777216.0 * norm;
    double cdf = 0;
    int chosen = static_cast<int>(vocab - 1);
    for (int64_t j = 0; j < vocab; ++j) {
      cdf += p[j];
      if (u < cdf) {
        chosen = static_cast<int>(j);
        break;
      }
    }
    out[i] = chosen;
  }
  return SamplerOutput{std::move(sampled), std::nullopt};
}

}  // namespace inferx::sampling
