#include "inferx/kernels/gpt_oss.h"

#include <cmath>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kBlock = 256;

__device__ inline float ToF32(bf16 x) { return __bfloat162float(x); }
__device__ inline bf16 ToBf16(float x) { return __float2bfloat16(x); }

// ln 2, for converting FlashInfer's base-2 lse into the natural log the sigmoid
// below wants.
constexpr float kLn2 = 0.6931471805599453f;

// One block per (token, head). The whole computation is one sigmoid and a
// scale over `head_dim` values, so the launch is most of the cost -- which is
// fine, because the alternative was patching a submodule.
__global__ void ApplySinksKernel(bf16* __restrict__ out,
                                 const float* __restrict__ lse,
                                 const bf16* __restrict__ sinks, int64_t heads,
                                 int64_t head_dim, float lse_to_natural) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;

  const float lse_nat = lse[token * heads + head] * lse_to_natural;
  const float sink = ToF32(sinks[head]);

  // sigmoid(lse - sink) = D / (D + e^sink), the fraction of the softmax mass
  // that the real keys hold once the sink is in the denominator. Written as a
  // sigmoid rather than as that ratio because the ratio overflows for a large
  // lse and the sigmoid does not.
  const float factor = 1.0f / (1.0f + __expf(sink - lse_nat));

  bf16* row = out + (token * heads + head) * head_dim;

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    row[d] = ToBf16(ToF32(row[d]) * factor);
  }
}

// One block per (query, head). Two passes over the visible keys: the maximum
// score, then the denominator and the weighted V together. Same shape as the
// other reference attentions here, and for the same reason -- it is the thing
// the fast path gets checked against.
__global__ void AttentionRefKernel(const bf16* __restrict__ q,
                                   const bf16* __restrict__ k,
                                   const bf16* __restrict__ v,
                                   bf16* __restrict__ out,
                                   float* __restrict__ lse, int64_t q_heads,
                                   int64_t kv_heads, int64_t head_dim,
                                   int64_t window, float scale) {
  const int64_t query = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t kv_head = head / (q_heads / kv_heads);

  // Causal, and windowed when asked. `first` is the oldest visible key: a
  // window of w lets query i see (i-w, i], which is w keys including itself.
  const int64_t last = query;
  const int64_t first =
      window > 0 ? (query - window + 1 > 0 ? query - window + 1 : 0) : 0;

  const bf16* qrow = q + (query * q_heads + head) * head_dim;

  __shared__ float reduce[kBlock];
  extern __shared__ float acc[];  // head_dim floats

  // Pass 1: the row maximum.
  float local_max = -INFINITY;
  for (int64_t j = first + threadIdx.x; j <= last; j += blockDim.x) {
    float dot = 0.0f;
    const bf16* krow = k + (j * kv_heads + kv_head) * head_dim;
    for (int64_t d = 0; d < head_dim; ++d) dot += ToF32(qrow[d]) * ToF32(krow[d]);
    local_max = fmaxf(local_max, dot * scale);
  }

  reduce[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      reduce[threadIdx.x] = fmaxf(reduce[threadIdx.x], reduce[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float row_max = reduce[0];
  __syncthreads();

  // Pass 2: denominator and weighted V.
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) acc[d] = 0.0f;
  __syncthreads();

  float local_sum = 0.0f;

  for (int64_t j = first; j <= last; ++j) {
    float dot = 0.0f;
    const bf16* krow = k + (j * kv_heads + kv_head) * head_dim;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      dot += ToF32(qrow[d]) * ToF32(krow[d]);
    }

    reduce[threadIdx.x] = dot;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
      __syncthreads();
    }

    const float weight = __expf(reduce[0] * scale - row_max);
    __syncthreads();

    if (threadIdx.x == 0) local_sum += weight;

    const bf16* vrow = v + (j * kv_heads + kv_head) * head_dim;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      acc[d] += weight * ToF32(vrow[d]);
    }
    __syncthreads();
  }

  __shared__ float denom;
  if (threadIdx.x == 0) denom = local_sum;
  __syncthreads();

  const float inv = 1.0f / denom;
  bf16* dst = out + (query * q_heads + head) * head_dim;
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    dst[d] = ToBf16(acc[d] * inv);
  }

  // FlashInfer's convention: m + log2(d), with m already in log2 space. Ours
  // is natural, so both terms convert. Writing it in *their* units is what
  // lets ApplyAttentionSinks be indifferent to which kernel produced it.
  if (threadIdx.x == 0) {
    lse[query * q_heads + head] =
        row_max * 1.4426950408889634f + __log2f(denom);
  }
}

__global__ void SwiGluKernel(const bf16* __restrict__ gate_up,
                             bf16* __restrict__ out, int64_t inter, float limit,
                             float alpha) {
  const int64_t token = blockIdx.x;
  const bf16* row = gate_up + token * 2 * inter;
  bf16* dst = out + token * inter;

  for (int64_t j = threadIdx.x; j < inter; j += blockDim.x) {
    // One-sided on the gate, two-sided on up: that asymmetry is gpt-oss's, not
    // a transcription slip. HF clamps gate with max= only.
    float gate = fminf(ToF32(row[j]), limit);
    float up = fmaxf(fminf(ToF32(row[inter + j]), limit), -limit);

    const float glu = gate / (1.0f + __expf(-gate * alpha));

    dst[j] = ToBf16((up + 1.0f) * glu);
  }
}

__global__ void RopeTableKernel(bf16* __restrict__ q, bf16* __restrict__ k,
                                const int32_t* __restrict__ positions,
                                const float* __restrict__ inv_freq,
                                int64_t q_heads, int64_t kv_heads,
                                int64_t head_dim, float attn_factor) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t half = head_dim / 2;
  const float pos = static_cast<float>(positions[token]);

  for (int64_t j = threadIdx.x; j < half; j += blockDim.x) {
    const float angle = pos * inv_freq[j];

    float s, c;
    sincosf(angle, &s, &c);

    // YaRN's temperature. 1.0 collapses this to the ordinary rotation, which
    // is why the same kernel serves both.
    s *= attn_factor;
    c *= attn_factor;

    if (head < q_heads) {
      bf16* row = q + (token * q_heads + head) * head_dim;
      const float lo = ToF32(row[j]);
      const float hi = ToF32(row[j + half]);
      row[j] = ToBf16(lo * c - hi * s);
      row[j + half] = ToBf16(hi * c + lo * s);
    }

    if (head < kv_heads) {
      bf16* row = k + (token * kv_heads + head) * head_dim;
      const float lo = ToF32(row[j]);
      const float hi = ToF32(row[j + half]);
      row[j] = ToBf16(lo * c - hi * s);
      row[j + half] = ToBf16(hi * c + lo * s);
    }
  }
}

Status CheckTensor(const TensorView& t, DataType dtype, int rank,
                   const char* name) {
  if (!t.IsDefined()) return InvalidArgumentError(name, " is undefined");

  if (!t.IsCuda()) {
    return InvalidArgumentError(name, " is on ", t.Device().ToString(),
                                ", not a CUDA device");
  }

  if (t.GetDataType() != dtype) {
    return InvalidArgumentError(name, " is ", DataTypeName(t.GetDataType()),
                                ", expected ", DataTypeName(dtype));
  }

  if (t.Rank() != rank) {
    return InvalidArgumentError(name, " has rank ", t.Rank(), ", expected ",
                                rank);
  }

  return OkStatus();
}

}  // namespace

Status ApplyAttentionSinks(const TensorView& out, const TensorView& lse,
                           const TensorView& sinks, bool lse_is_log2,
                           cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckTensor(lse, DataType::kFloat, 2, "lse"));
  INFERX_RETURN_IF_ERROR(CheckTensor(sinks, DataType::kBFloat16, 1, "sinks"));

  const int64_t tokens = out.Dim(0);
  const int64_t heads = out.Dim(1);
  const int64_t head_dim = out.Dim(2);

  if (lse.Dim(0) != tokens || lse.Dim(1) != heads) {
    return InvalidArgumentError("lse must be [", tokens, ", ", heads,
                                "], got ", lse.GetShape().ToString());
  }

  if (sinks.Dim(0) != heads) {
    return InvalidArgumentError("sinks must hold one logit per head (", heads,
                                "), got ", sinks.Dim(0));
  }

  if (tokens == 0) return OkStatus();

  const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(heads));

  ApplySinksKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<bf16*>(out.Data()), static_cast<const float*>(lse.Data()),
      static_cast<const bf16*>(sinks.Data()), heads, head_dim,
      lse_is_log2 ? kLn2 : 1.0f);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status GptOssAttentionRef(const TensorView& q, const TensorView& k,
                          const TensorView& v, const TensorView& out,
                          const TensorView& lse, int64_t window, float scale,
                          cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(q, DataType::kBFloat16, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k, DataType::kBFloat16, 3, "k"));
  INFERX_RETURN_IF_ERROR(CheckTensor(v, DataType::kBFloat16, 3, "v"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckTensor(lse, DataType::kFloat, 2, "lse"));

  const int64_t tokens = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t kv_heads = k.Dim(1);

  if (k.Dim(0) != tokens || v.Dim(0) != tokens || k.Dim(2) != head_dim ||
      v.Dim(2) != head_dim || v.Dim(1) != kv_heads) {
    return InvalidArgumentError("k ", k.GetShape().ToString(), " and v ",
                                v.GetShape().ToString(), " do not match q ",
                                q.GetShape().ToString());
  }

  if (kv_heads <= 0 || q_heads % kv_heads != 0) {
    return InvalidArgumentError("q_heads ", q_heads,
                                " is not a multiple of kv_heads ", kv_heads);
  }

  if (out.Dim(0) != tokens || out.Dim(1) != q_heads || out.Dim(2) != head_dim) {
    return InvalidArgumentError("out is ", out.GetShape().ToString(),
                                " but q is ", q.GetShape().ToString());
  }

  if (lse.Dim(0) != tokens || lse.Dim(1) != q_heads) {
    return InvalidArgumentError("lse must be [", tokens, ", ", q_heads,
                                "], got ", lse.GetShape().ToString());
  }

  if (window < 0) {
    return InvalidArgumentError("window must be non-negative, got ", window);
  }

  if (tokens == 0) return OkStatus();

  const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(q_heads));
  const size_t shared = static_cast<size_t>(head_dim) * sizeof(float);

  AttentionRefKernel<<<grid, kBlock, shared, stream>>>(
      static_cast<const bf16*>(q.Data()), static_cast<const bf16*>(k.Data()),
      static_cast<const bf16*>(v.Data()), static_cast<bf16*>(out.Data()),
      static_cast<float*>(lse.Data()), q_heads, kv_heads, head_dim, window,
      scale);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status GptOssSwiGlu(const TensorView& gate_up, const TensorView& out,
                    float limit, float alpha, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(
      CheckTensor(gate_up, DataType::kBFloat16, 2, "gate_up"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));

  const int64_t tokens = out.Dim(0);
  const int64_t inter = out.Dim(1);

  if (gate_up.Dim(0) != tokens || gate_up.Dim(1) != 2 * inter) {
    return InvalidArgumentError("gate_up must be [", tokens, ", ", 2 * inter,
                                "], got ", gate_up.GetShape().ToString());
  }

  if (limit <= 0.0f) {
    return InvalidArgumentError("swiglu limit must be positive, got ", limit);
  }

  if (tokens == 0) return OkStatus();

  SwiGluKernel<<<static_cast<int>(tokens), kBlock, 0, stream>>>(
      static_cast<const bf16*>(gate_up.Data()), static_cast<bf16*>(out.Data()),
      inter, limit, alpha);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status RotaryEmbeddingFromTable(const TensorView& q, const TensorView& k,
                                const TensorView& positions,
                                const TensorView& inv_freq, float attn_factor,
                                cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(q, DataType::kBFloat16, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k, DataType::kBFloat16, 3, "k"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(positions, DataType::kInt32, 1, "positions"));
  INFERX_RETURN_IF_ERROR(CheckTensor(inv_freq, DataType::kFloat, 1, "inv_freq"));

  const int64_t tokens = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t kv_heads = k.Dim(1);

  if (k.Dim(0) != tokens || k.Dim(2) != head_dim) {
    return InvalidArgumentError("k is ", k.GetShape().ToString(), " but q is ",
                                q.GetShape().ToString());
  }

  if (positions.Dim(0) != tokens) {
    return InvalidArgumentError("positions has ", positions.Dim(0),
                                " entries but there are ", tokens, " tokens");
  }

  if (head_dim % 2 != 0) {
    return InvalidArgumentError("head_dim must be even for RoPE, got ",
                                head_dim);
  }

  if (inv_freq.Dim(0) != head_dim / 2) {
    return InvalidArgumentError("inv_freq must hold ", head_dim / 2,
                                " entries, got ", inv_freq.Dim(0));
  }

  if (tokens == 0) return OkStatus();

  const dim3 grid(static_cast<unsigned>(tokens),
                  static_cast<unsigned>(q_heads > kv_heads ? q_heads : kv_heads));

  RopeTableKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<bf16*>(q.Data()), static_cast<bf16*>(k.Data()),
      static_cast<const int32_t*>(positions.Data()),
      static_cast<const float*>(inv_freq.Data()), q_heads, kv_heads, head_dim,
      attn_factor);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

float ComputeYarnInvFreq(int64_t head_dim, double base, double factor,
                         double beta_fast, double beta_slow,
                         int64_t original_max, bool truncate, float* out) {
  const int64_t half = head_dim / 2;

  // The dimension at which a given number of rotations fits in the original
  // context. Inverted from `rot = original_max / (2 pi / inv_freq_d)`.
  auto find_correction_dim = [&](double rotations) {
    return (static_cast<double>(head_dim) *
            std::log(static_cast<double>(original_max) /
                     (rotations * 2.0 * M_PI))) /
           (2.0 * std::log(base));
  };

  double low = find_correction_dim(beta_fast);
  double high = find_correction_dim(beta_slow);

  // `truncate` selects whether the range snaps to whole dimensions. The clamp
  // to [0, head_dim-1] is *not* part of it and applies either way -- a detail
  // worth being careful about, because gpt-oss sets truncate false and the two
  // are easy to conflate into one branch.
  if (truncate) {
    low = std::floor(low);
    high = std::ceil(high);
  }

  low = std::max(low, 0.0);
  high = std::min(high, static_cast<double>(head_dim) - 1.0);

  for (int64_t j = 0; j < half; ++j) {
    const double exponent = 2.0 * static_cast<double>(j) /
                            static_cast<double>(head_dim);
    const double pos_freq = std::pow(base, exponent);

    const double extrapolation = 1.0 / pos_freq;       // the original frequency
    const double interpolation = 1.0 / (factor * pos_freq);  // stretched

    // A linear ramp between the two correction dimensions: low-frequency
    // components interpolate (they must, to reach the longer context), and
    // high-frequency ones extrapolate (they must, to keep local detail).
    // A degenerate window would divide by zero; HF widens it by 0.001, and
    // matching that exactly matters more than it being elegant.
    const double span = (high == low) ? 0.001 : (high - low);
    double ramp = (static_cast<double>(j) - low) / span;
    ramp = std::min(std::max(ramp, 0.0), 1.0);

    const double extrapolation_factor = 1.0 - ramp;

    out[j] = static_cast<float>(
        interpolation * (1.0 - extrapolation_factor) +
        extrapolation * extrapolation_factor);
  }

  // YaRN's attention temperature. Reached by default rather than by
  // configuration for gpt-oss, and ~1.34657 at factor 32.
  return YarnMscale(factor, 1.0);
}

float YarnMscale(double factor, double mscale) {
  if (factor <= 1.0) return 1.0f;
  return static_cast<float>(0.1 * mscale * std::log(factor) + 1.0);
}

}  // namespace inferx::kernels
