#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>

#include <algorithm>
#include <utility>

#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/ops/layers.h"

namespace inferx::ops {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kWarp = 32;

__device__ inline float ToF32(bf16 x) { return __bfloat162float(x); }
__device__ inline bf16 ToBf16(float x) { return __float2bfloat16(x); }

// Sums `v` across the block via shared memory. `tile` must hold blockDim.x
// floats. Every thread gets the total.
template <typename Op>
__device__ float BlockReduce(float v, float* tile, Op op, float identity) {
  const int tid = threadIdx.x;

  tile[tid] = v;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (tid < stride) tile[tid] = op(tile[tid], tile[tid + stride]);
    __syncthreads();
  }

  const float total = tile[0];
  __syncthreads();  // before any caller reuses the tile

  (void)identity;
  return total;
}

struct SumOp {
  __device__ float operator()(float a, float b) const { return a + b; }
};

// One block per token. hidden is a multiple of the block size in every model we
// target, but the loops are written as grid-strides so an odd width still
// works.
__global__ void RmsNormKernel(const bf16* __restrict__ x,
                              const bf16* __restrict__ weight,
                              bf16* __restrict__ out, int64_t hidden,
                              float eps) {
  extern __shared__ float tile[];

  const int64_t row = blockIdx.x;
  const bf16* xr = x + row * hidden;
  bf16* outr = out + row * hidden;

  float sumsq = 0.0f;
  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    const float v = ToF32(xr[i]);
    sumsq += v * v;
  }

  sumsq = BlockReduce(sumsq, tile, SumOp{}, 0.0f);

  // rsqrtf of the fp32 mean. Computed once per block and broadcast rather than
  // per element, which also keeps every element scaled by bitwise the same
  // factor -- a per-thread recomputation would not be guaranteed to.
  const float inv = rsqrtf(sumsq / static_cast<float>(hidden) + eps);

  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) {
    outr[i] = ToBf16(ToF32(xr[i]) * inv * ToF32(weight[i]));
  }
}

// One block per (token, head); each thread owns one index below head_dim/2 and
// rotates the pair (j, j + half) together.
__global__ void RopeKernel(bf16* __restrict__ q, bf16* __restrict__ k,
                           const int32_t* __restrict__ positions,
                           int64_t q_heads, int64_t kv_heads, int64_t head_dim,
                           float theta) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t half = head_dim / 2;
  const float pos = static_cast<float>(positions[token]);

  for (int64_t j = threadIdx.x; j < half; j += blockDim.x) {
    // inv_freq[j] = theta^(-2j/d). __powf is fine here: the exponent is small
    // and the result feeds a sinf/cosf whose own error dominates.
    const float inv_freq = __powf(
        theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    const float angle = pos * inv_freq;

    float s, c;
    __sincosf(angle, &s, &c);

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

__global__ void SiluMulKernel(const bf16* __restrict__ gate,
                              const bf16* __restrict__ up,
                              bf16* __restrict__ out, int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const float g = ToF32(gate[i]);
    // silu(g) = g / (1 + e^-g). Computed in fp32; in bf16 the sigmoid would
    // quantize to about 256 distinct values over its useful range.
    const float silu = g / (1.0f + __expf(-g));
    out[i] = ToBf16(silu * ToF32(up[i]));
  }
}

// One block per (query token, query head). Two passes over the keys: one for
// the max and sum, one to accumulate V. Simple and obviously correct, which is
// what this kernel is for.
__global__ void AttentionKernel(const bf16* __restrict__ q,
                                const bf16* __restrict__ k,
                                const bf16* __restrict__ v,
                                bf16* __restrict__ out, int64_t tokens,
                                int64_t q_heads, int64_t kv_heads,
                                int64_t head_dim, float scale) {
  extern __shared__ float smem[];
  float* tile = smem;                 // blockDim.x floats for reductions
  float* scores = smem + blockDim.x;  // one float per key

  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t group = q_heads / kv_heads;
  const int64_t kv_head = head / group;  // GQA, without materializing repeats

  const bf16* qr = q + (token * q_heads + head) * head_dim;

  // Causal: a query at position `token` sees keys 0..token inclusive.
  const int64_t n_keys = token + 1;

  float running_max = -INFINITY;

  for (int64_t j = 0; j < n_keys; ++j) {
    const bf16* kr = k + (j * kv_heads + kv_head) * head_dim;

    float dot = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      dot += ToF32(qr[d]) * ToF32(kr[d]);
    }
    dot = BlockReduce(dot, tile, SumOp{}, 0.0f);

    if (threadIdx.x == 0) scores[j] = dot * scale;
    __syncthreads();

    running_max = fmaxf(running_max, scores[j]);
  }

  // Softmax in fp32 with the max subtracted. The subtraction is not optional:
  // scores reach a few hundred at these widths and expf would overflow.
  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < n_keys; j += blockDim.x) {
    const float e = __expf(scores[j] - running_max);
    scores[j] = e;
    sum += e;
  }
  sum = BlockReduce(sum, tile, SumOp{}, 0.0f);

  const float inv_sum = 1.0f / sum;

  // Accumulate V. Each thread owns a slice of head_dim and walks every key.
  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t j = 0; j < n_keys; ++j) {
      const bf16* vr = v + (j * kv_heads + kv_head) * head_dim;
      acc += scores[j] * ToF32(vr[d]);
    }
    out[(token * q_heads + head) * head_dim + d] = ToBf16(acc * inv_sum);
  }
}

// One block per (token, kv_head). The slot is precomputed host-side, so this
// kernel never sees a block table and does not care how blocks were assigned.
__global__ void AppendKvKernel(const bf16* __restrict__ k,
                               const bf16* __restrict__ v,
                               bf16* __restrict__ k_cache,
                               bf16* __restrict__ v_cache,
                               const int32_t* __restrict__ slots,
                               int64_t kv_heads, int64_t head_dim,
                               int64_t total_slots) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t slot = slots[token];

  // A bad slot would scatter a token's keys into another sequence's cache,
  // which surfaces as garbled output from an unrelated request. Dropped rather
  // than clamped: writing to slot 0 would be silent corruption, and the host
  // side validates before launch anyway.
  if (slot < 0 || slot >= total_slots) return;

  const int64_t src = (token * kv_heads + head) * head_dim;
  const int64_t dst = (slot * kv_heads + head) * head_dim;

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    k_cache[dst + d] = k[src + d];
    v_cache[dst + d] = v[src + d];
  }
}

// Same scatter structure as AppendKvKernel -- one block per (token, kv_head),
// slot precomputed host-side -- but it quantizes bf16 -> fp8 e4m3 against a
// frozen per-layer scale as it writes. Fused rather than quantize-then-scatter
// so the fp8 KV write stays one pass over K/V: the decode step is bandwidth-
// bound, and a separate fp8 scratch the scatter would then read back doubles
// the K/V traffic for nothing. The scale is fixed at warmup, so baking it into
// the kernel argument pins a stable value for any captured decode graph.
__global__ void AppendBf16AsFp8Kernel(const bf16* __restrict__ k,
                                      const bf16* __restrict__ v,
                                      __nv_fp8_storage_t* __restrict__ k_cache,
                                      __nv_fp8_storage_t* __restrict__ v_cache,
                                      const int32_t* __restrict__ slots,
                                      int64_t kv_heads, int64_t head_dim,
                                      int64_t total_slots, float inv_k_scale,
                                      float inv_v_scale) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t slot = slots[token];

  if (slot < 0 || slot >= total_slots) return;

  const int64_t src = (token * kv_heads + head) * head_dim;
  const int64_t dst = (slot * kv_heads + head) * head_dim;

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    k_cache[dst + d] = __nv_cvt_float_to_fp8(
        __bfloat162float(k[src + d]) * inv_k_scale, __NV_SATFINITE, __NV_E4M3);
    v_cache[dst + d] = __nv_cvt_float_to_fp8(
        __bfloat162float(v[src + d]) * inv_v_scale, __NV_SATFINITE, __NV_E4M3);
  }
}

// One block per (query token, query head), same as the contiguous kernel. The
// only difference is where a key comes from: `block_table[seq][j / block_size]`
// gives the block, `j % block_size` the slot inside it.
__global__ void PagedAttentionKernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
    const bf16* __restrict__ v_cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_of_token, const int32_t* __restrict__ q_pos,
    bf16* __restrict__ out, int64_t q_heads, int64_t kv_heads, int64_t head_dim,
    int64_t block_size, int64_t max_blocks, float scale, int64_t max_keys) {
  extern __shared__ float smem[];
  float* tile = smem;
  float* scores = smem + blockDim.x;

  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t seq = seq_of_token[token];
  const int64_t group = q_heads / kv_heads;
  const int64_t kv_head = head / group;

  // Causal: a query at absolute position p attends to keys 0..p.
  const int64_t n_keys = static_cast<int64_t>(q_pos[token]) + 1;
  if (n_keys <= 0 || n_keys > max_keys) return;

  const bf16* qr = q + (token * q_heads + head) * head_dim;
  const int32_t* table = block_table + seq * max_blocks;

  float running_max = -INFINITY;

  for (int64_t j = 0; j < n_keys; ++j) {
    const int32_t block = table[j / block_size];
    const int64_t slot = block * block_size + (j % block_size);
    const bf16* kr = k_cache + (slot * kv_heads + kv_head) * head_dim;

    float dot = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      dot += ToF32(qr[d]) * ToF32(kr[d]);
    }
    dot = BlockReduce(dot, tile, SumOp{}, 0.0f);

    if (threadIdx.x == 0) scores[j] = dot * scale;
    __syncthreads();

    running_max = fmaxf(running_max, scores[j]);
  }

  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < n_keys; j += blockDim.x) {
    const float e = __expf(scores[j] - running_max);
    scores[j] = e;
    sum += e;
  }
  sum = BlockReduce(sum, tile, SumOp{}, 0.0f);

  const float inv_sum = 1.0f / sum;

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t j = 0; j < n_keys; ++j) {
      const int32_t block = table[j / block_size];
      const int64_t slot = block * block_size + (j % block_size);
      acc += scores[j] *
             ToF32(v_cache[(slot * kv_heads + kv_head) * head_dim + d]);
    }
    out[(token * q_heads + head) * head_dim + d] = ToBf16(acc * inv_sum);
  }
}

// PagedAttentionKernel with two gpt-oss additions: a per-(token, head)
// log-sum-exp output (the ingredient that makes a sink a post-pass rescale
// rather than a kernel change) and a sliding window (gpt-oss alternates full
// and 128-token layers). Otherwise identical arithmetic, kept separate from the
// plain kernel so the Qwen2 serving path -- which wants neither -- runs exactly
// the launch it always did.
//
// `lse` is written in the natural-log convention (`max + ln(sum)`); pair with
// `ApplyAttentionSinks(..., /*lse_is_log2=*/false)`.
__global__ void PagedAttentionWithLseKernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
    const bf16* __restrict__ v_cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_of_token, const int32_t* __restrict__ q_pos,
    bf16* __restrict__ out, float* __restrict__ lse, int64_t q_heads,
    int64_t kv_heads, int64_t head_dim, int64_t block_size, int64_t max_blocks,
    float scale, int64_t window, int64_t max_keys) {
  extern __shared__ float smem[];
  float* tile = smem;
  float* scores = smem + blockDim.x;

  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t seq = seq_of_token[token];
  const int64_t group = q_heads / kv_heads;
  const int64_t kv_head = head / group;

  // A query at absolute position p sees keys [start_key, p]. With no window
  // that is 0..p (the plain causal mask); with a window of W it is max(0,
  // p-W+1)..p, so the oldest keys fall out of view rather than out of memory --
  // the cache still holds them, this kernel just stops reading them.
  const int64_t pos = static_cast<int64_t>(q_pos[token]);
  const int64_t back = (window > 0 && pos + 1 > window) ? pos + 1 - window : 0;
  const int64_t start_key = back;
  const int64_t n_keys = pos + 1 - start_key;
  if (n_keys <= 0 || n_keys > max_keys) return;

  const bf16* qr = q + (token * q_heads + head) * head_dim;
  const int32_t* table = block_table + seq * max_blocks;

  float running_max = -INFINITY;

  for (int64_t j = 0; j < n_keys; ++j) {
    const int64_t key = start_key + j;
    const int32_t block = table[key / block_size];
    const int64_t slot = block * block_size + (key % block_size);
    const bf16* kr = k_cache + (slot * kv_heads + kv_head) * head_dim;

    float dot = 0.0f;
    for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
      dot += ToF32(qr[d]) * ToF32(kr[d]);
    }
    dot = BlockReduce(dot, tile, SumOp{}, 0.0f);

    if (threadIdx.x == 0) scores[j] = dot * scale;
    __syncthreads();

    running_max = fmaxf(running_max, scores[j]);
  }

  float sum = 0.0f;
  for (int64_t j = threadIdx.x; j < n_keys; j += blockDim.x) {
    const float e = __expf(scores[j] - running_max);
    scores[j] = e;
    sum += e;
  }
  sum = BlockReduce(sum, tile, SumOp{}, 0.0f);

  const float inv_sum = 1.0f / sum;

  // lse = max + ln(sum). Written once per (token, head); `sum` is broadcast by
  // BlockReduce so the value is identical across threads, but the write is
  // thread 0 to avoid a race.
  if (lse != nullptr && threadIdx.x == 0) {
    lse[token * q_heads + head] = running_max + logf(sum);
  }

  for (int64_t d = threadIdx.x; d < head_dim; d += blockDim.x) {
    float acc = 0.0f;
    for (int64_t j = 0; j < n_keys; ++j) {
      const int64_t key = start_key + j;
      const int32_t block = table[key / block_size];
      const int64_t slot = block * block_size + (key % block_size);
      acc += scores[j] *
             ToF32(v_cache[(slot * kv_heads + kv_head) * head_dim + d]);
    }
    out[(token * q_heads + head) * head_dim + d] = ToBf16(acc * inv_sum);
  }
}

// One block per row. Each thread scans a strided slice of the vocabulary and
// the block reduces to a single (value, index) pair. Ties go to the lower index
// so the result is deterministic, which matters because two runs of the same
// prompt must produce the same token.
__global__ void ArgmaxKernel(const bf16* __restrict__ logits,
                             const int32_t* __restrict__ rows,
                             int32_t* __restrict__ out, int64_t vocab) {
  __shared__ float best_val[256];
  __shared__ int32_t best_idx[256];

  const int64_t which = blockIdx.x;
  const bf16* r = logits + static_cast<int64_t>(rows[which]) * vocab;

  float v = -INFINITY;
  int32_t i = 0;

  for (int64_t j = threadIdx.x; j < vocab; j += blockDim.x) {
    const float x = ToF32(r[j]);
    if (x > v) {
      v = x;
      i = static_cast<int32_t>(j);
    }
  }

  best_val[threadIdx.x] = v;
  best_idx[threadIdx.x] = i;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      const int other = threadIdx.x + stride;
      const bool take = best_val[other] > best_val[threadIdx.x] ||
                        (best_val[other] == best_val[threadIdx.x] &&
                         best_idx[other] < best_idx[threadIdx.x]);
      if (take) {
        best_val[threadIdx.x] = best_val[other];
        best_idx[threadIdx.x] = best_idx[other];
      }
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) out[which] = best_idx[0];
}

__global__ void ScatterTokensKernel(const int32_t* __restrict__ src,
                                    int32_t* __restrict__ dst,
                                    const int32_t* __restrict__ slots,
                                    int64_t n, int64_t m) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int32_t slot = slots[i];
    if (slot >= 0 && slot < m) dst[slot] = src[i];
  }
}

__global__ void EmbeddingKernel(const bf16* __restrict__ table,
                                const int32_t* __restrict__ ids,
                                bf16* __restrict__ out, int64_t hidden,
                                int64_t vocab) {
  const int64_t token = blockIdx.x;
  const int32_t id = ids[token];

  // Clamped rather than trusted. An out-of-range id would read arbitrary memory
  // from the embedding table; the host-side check rejects it before launch, and
  // this is the belt to that braces.
  const int64_t row = (id < 0 || id >= vocab) ? 0 : id;

  const bf16* src = table + row * hidden;
  bf16* dst = out + token * hidden;

  for (int64_t i = threadIdx.x; i < hidden; i += blockDim.x) dst[i] = src[i];
}

// One block per (token, destination). Each thread copies one element from the
// fused row into whichever of Q/K/V owns it, adding the bias for that column.
__global__ void SplitQkvKernel(const bf16* __restrict__ fused,
                               const bf16* __restrict__ bias,
                               bf16* __restrict__ q, bf16* __restrict__ k,
                               bf16* __restrict__ v, int64_t tokens,
                               int64_t q_dim, int64_t kv_dim) {
  const int64_t width = q_dim + 2 * kv_dim;
  const int64_t total = tokens * width;

  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t token = idx / width;
    const int64_t col = idx % width;

    float value = ToF32(fused[idx]);
    if (bias != nullptr) value += ToF32(bias[col]);

    if (col < q_dim) {
      q[token * q_dim + col] = ToBf16(value);
    } else if (col < q_dim + kv_dim) {
      k[token * kv_dim + (col - q_dim)] = ToBf16(value);
    } else {
      v[token * kv_dim + (col - q_dim - kv_dim)] = ToBf16(value);
    }
  }
}

__global__ void SiluMulFusedKernel(const bf16* __restrict__ fused,
                                   bf16* __restrict__ out, int64_t tokens,
                                   int64_t inter) {
  const int64_t total = tokens * inter;

  for (int64_t idx =
           static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       idx < total; idx += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t token = idx / inter;
    const int64_t col = idx % inter;

    // gate occupies the first half of each row, up the second.
    const float g = ToF32(fused[token * 2 * inter + col]);
    const float u = ToF32(fused[token * 2 * inter + inter + col]);

    out[idx] = ToBf16((g / (1.0f + __expf(-g))) * u);
  }
}

__global__ void AddBiasKernel(bf16* __restrict__ out,
                              const bf16* __restrict__ bias, int64_t tokens,
                              int64_t width) {
  const int64_t total = tokens * width;

  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < total; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = ToBf16(ToF32(out[i]) + ToF32(bias[i % width]));
  }
}

__global__ void AddKernel(bf16* __restrict__ out, const bf16* __restrict__ rhs,
                          int64_t n) {
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       i < n; i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    out[i] = ToBf16(ToF32(out[i]) + ToF32(rhs[i]));
  }
}

// ---------------------------------------------------------------------------
// Host-side validation. Shared because every kernel below wants the same three
// questions answered, and a wrong answer is an out-of-bounds device read.
// ---------------------------------------------------------------------------

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

Status CheckSameShape(const TensorView& a, const TensorView& b, const char* an,
                      const char* bn) {
  if (a.Rank() != b.Rank()) {
    return InvalidArgumentError(an, " has rank ", a.Rank(), " but ", bn,
                                " has rank ", b.Rank());
  }

  for (int i = 0; i < a.Rank(); ++i) {
    if (a.Dim(i) != b.Dim(i)) {
      return InvalidArgumentError(an, " is ", a.GetShape().ToString(), " but ",
                                  bn, " is ", b.GetShape().ToString());
    }
  }

  return OkStatus();
}

int BlockFor(int64_t width) {
  // Powers of two only: BlockReduce halves the block each step and would drop
  // the odd element otherwise.
  int block = kWarp;
  while (block < width && block < 1024) block *= 2;
  return block;
}

}  // namespace

Status RmsNorm(const TensorView& x, const TensorView& weight,
               const TensorView& out, float eps, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(x, DataType::kBFloat16, 2, "x"));
  INFERX_RETURN_IF_ERROR(CheckTensor(weight, DataType::kBFloat16, 1, "weight"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(x, out, "x", "out"));

  const int64_t tokens = x.Dim(0);
  const int64_t hidden = x.Dim(1);

  if (weight.Dim(0) != hidden) {
    return InvalidArgumentError("weight has ", weight.Dim(0),
                                " elements but "
                                "hidden is ",
                                hidden);
  }

  if (tokens == 0) return OkStatus();

  const int block = BlockFor(hidden);

  RmsNormKernel<<<static_cast<unsigned>(tokens), block, block * sizeof(float),
                  stream>>>(static_cast<const bf16*>(x.Data()),
                            static_cast<const bf16*>(weight.Data()),
                            static_cast<bf16*>(out.Data()), hidden, eps);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status RotaryEmbedding(const TensorView& q, const TensorView& k,
                       const TensorView& positions, float theta,
                       Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(q, DataType::kBFloat16, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k, DataType::kBFloat16, 3, "k"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(positions, DataType::kInt32, 1, "positions"));

  const int64_t tokens = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t kv_heads = k.Dim(1);

  if (k.Dim(0) != tokens) {
    return InvalidArgumentError("q has ", tokens, " tokens but k has ",
                                k.Dim(0));
  }
  if (k.Dim(2) != head_dim) {
    return InvalidArgumentError("q head_dim is ", head_dim, " but k's is ",
                                k.Dim(2));
  }
  if (positions.Dim(0) != tokens) {
    return InvalidArgumentError("positions has ", positions.Dim(0),
                                " entries but there are ", tokens, " tokens");
  }
  if (head_dim % 2 != 0) {
    return InvalidArgumentError("head_dim must be even for RoPE, got ",
                                head_dim);
  }

  if (tokens == 0) return OkStatus();

  const int64_t heads = q_heads > kv_heads ? q_heads : kv_heads;
  const int block = BlockFor(head_dim / 2);

  RopeKernel<<<dim3(static_cast<unsigned>(tokens),
                    static_cast<unsigned>(heads)),
               block, 0, stream>>>(
      static_cast<bf16*>(q.Data()), static_cast<bf16*>(k.Data()),
      static_cast<const int32_t*>(positions.Data()), q_heads, kv_heads,
      head_dim, theta);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status SiluMul(const TensorView& gate, const TensorView& up,
               const TensorView& out, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(gate, DataType::kBFloat16, 2, "gate"));
  INFERX_RETURN_IF_ERROR(CheckTensor(up, DataType::kBFloat16, 2, "up"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(gate, up, "gate", "up"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(gate, out, "gate", "out"));

  const int64_t n = gate.Numel();
  if (n == 0) return OkStatus();

  constexpr int kBlock = 256;
  const int64_t grid_want = (n + kBlock - 1) / kBlock;
  const unsigned grid =
      static_cast<unsigned>(grid_want > 4096 ? 4096 : grid_want);

  SiluMulKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const bf16*>(gate.Data()),
      static_cast<const bf16*>(up.Data()), static_cast<bf16*>(out.Data()), n);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status Attention(const TensorView& q, const TensorView& k, const TensorView& v,
                 const TensorView& out, float scale, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(q, DataType::kBFloat16, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k, DataType::kBFloat16, 3, "k"));
  INFERX_RETURN_IF_ERROR(CheckTensor(v, DataType::kBFloat16, 3, "v"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(k, v, "k", "v"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(q, out, "q", "out"));

  const int64_t tokens = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t kv_heads = k.Dim(1);

  if (k.Dim(0) != tokens) {
    return InvalidArgumentError("q has ", tokens, " tokens but k has ",
                                k.Dim(0));
  }
  if (k.Dim(2) != head_dim) {
    return InvalidArgumentError("q head_dim is ", head_dim, " but k's is ",
                                k.Dim(2));
  }
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    return InvalidArgumentError("q_heads (", q_heads,
                                ") is not a multiple of "
                                "kv_heads (",
                                kv_heads, ")");
  }

  if (tokens == 0) return OkStatus();

  const int block = BlockFor(head_dim);

  // Shared memory holds the reduction tile plus one score per key. The score
  // array is sized for the longest query, which is the last token.
  const size_t smem =
      (static_cast<size_t>(block) + static_cast<size_t>(tokens)) *
      sizeof(float);

  // 48 KB is the default per-block limit without opting in to more. At bf16
  // this caps M2 at ~11k tokens, well past the 32k context we would need a
  // paged kernel for anyway (that is M3).
  if (smem > 48u * 1024u) {
    return ResourceExhaustedError(
        "naive attention needs ", smem, " B of shared memory for ", tokens,
        " tokens, over the 48 KB limit; this kernel is the M2 reference, use "
        "the paged path for long sequences");
  }

  AttentionKernel<<<dim3(static_cast<unsigned>(tokens),
                         static_cast<unsigned>(q_heads)),
                    block, smem, stream>>>(
      static_cast<const bf16*>(q.Data()), static_cast<const bf16*>(k.Data()),
      static_cast<const bf16*>(v.Data()), static_cast<bf16*>(out.Data()),
      tokens, q_heads, kv_heads, head_dim, scale);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status AppendToKvCache(const TensorView& k, const TensorView& v,
                       const TensorView& k_cache, const TensorView& v_cache,
                       const TensorView& slots, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(k, DataType::kBFloat16, 3, "k"));
  INFERX_RETURN_IF_ERROR(CheckTensor(v, DataType::kBFloat16, 3, "v"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(k_cache, DataType::kBFloat16, 4, "k_cache"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(v_cache, DataType::kBFloat16, 4, "v_cache"));
  INFERX_RETURN_IF_ERROR(CheckTensor(slots, DataType::kInt32, 1, "slots"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(k, v, "k", "v"));
  INFERX_RETURN_IF_ERROR(
      CheckSameShape(k_cache, v_cache, "k_cache", "v_cache"));

  const int64_t tokens = k.Dim(0);
  const int64_t kv_heads = k.Dim(1);
  const int64_t head_dim = k.Dim(2);

  if (slots.Dim(0) != tokens) {
    return InvalidArgumentError("slots has ", slots.Dim(0),
                                " entries but "
                                "there are ",
                                tokens, " tokens");
  }
  if (k_cache.Dim(2) != kv_heads || k_cache.Dim(3) != head_dim) {
    return InvalidArgumentError("cache is [.., .., ", k_cache.Dim(2), ", ",
                                k_cache.Dim(3), "] but k is [.., ", kv_heads,
                                ", ", head_dim, "]");
  }

  if (tokens == 0) return OkStatus();

  const int64_t total_slots = k_cache.Dim(0) * k_cache.Dim(1);
  const int block = BlockFor(head_dim);

  AppendKvKernel<<<dim3(static_cast<unsigned>(tokens),
                        static_cast<unsigned>(kv_heads)),
                   block, 0, stream>>>(
      static_cast<const bf16*>(k.Data()), static_cast<const bf16*>(v.Data()),
      static_cast<bf16*>(k_cache.Data()), static_cast<bf16*>(v_cache.Data()),
      static_cast<const int32_t*>(slots.Data()), kv_heads, head_dim,
      total_slots);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status AppendBf16AsFp8(const TensorView& k, const TensorView& v,
                       const TensorView& k_cache, const TensorView& v_cache,
                       const TensorView& slots, float k_scale, float v_scale,
                       Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(k, DataType::kBFloat16, 3, "k"));
  INFERX_RETURN_IF_ERROR(CheckTensor(v, DataType::kBFloat16, 3, "v"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(k_cache, DataType::kFloat8E4M3FN, 4, "k_cache"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(v_cache, DataType::kFloat8E4M3FN, 4, "v_cache"));
  INFERX_RETURN_IF_ERROR(CheckTensor(slots, DataType::kInt32, 1, "slots"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(k, v, "k", "v"));
  INFERX_RETURN_IF_ERROR(
      CheckSameShape(k_cache, v_cache, "k_cache", "v_cache"));

  if (k_scale <= 0.0f || v_scale <= 0.0f) {
    return InvalidArgumentError(
        "AppendBf16AsFp8: scales must be positive, got "
        "k_scale=",
        k_scale, " v_scale=", v_scale);
  }

  const int64_t tokens = k.Dim(0);
  const int64_t kv_heads = k.Dim(1);
  const int64_t head_dim = k.Dim(2);

  if (slots.Dim(0) != tokens) {
    return InvalidArgumentError("slots has ", slots.Dim(0),
                                " entries but there are ", tokens, " tokens");
  }
  if (k_cache.Dim(2) != kv_heads || k_cache.Dim(3) != head_dim) {
    return InvalidArgumentError("cache is [.., .., ", k_cache.Dim(2), ", ",
                                k_cache.Dim(3), "] but k is [.., ", kv_heads,
                                ", ", head_dim, "]");
  }

  if (tokens == 0) return OkStatus();

  const int64_t total_slots = k_cache.Dim(0) * k_cache.Dim(1);
  const int block = BlockFor(head_dim);

  AppendBf16AsFp8Kernel<<<dim3(static_cast<unsigned>(tokens),
                               static_cast<unsigned>(kv_heads)),
                          block, 0, stream>>>(
      static_cast<const bf16*>(k.Data()), static_cast<const bf16*>(v.Data()),
      static_cast<__nv_fp8_storage_t*>(k_cache.Data()),
      static_cast<__nv_fp8_storage_t*>(v_cache.Data()),
      static_cast<const int32_t*>(slots.Data()), kv_heads, head_dim,
      total_slots, 1.0f / k_scale, 1.0f / v_scale);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status PagedAttention(const TensorView& q, const TensorView& k_cache,
                      const TensorView& v_cache, const TensorView& block_table,
                      const TensorView& seq_of_token, const TensorView& q_pos,
                      const TensorView& out, float scale, int64_t max_context,
                      Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(q, DataType::kBFloat16, 3, "q"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(k_cache, DataType::kBFloat16, 4, "k_cache"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(v_cache, DataType::kBFloat16, 4, "v_cache"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(block_table, DataType::kInt32, 2, "block_table"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(seq_of_token, DataType::kInt32, 1, "seq_of_token"));
  INFERX_RETURN_IF_ERROR(CheckTensor(q_pos, DataType::kInt32, 1, "q_pos"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(q, out, "q", "out"));
  INFERX_RETURN_IF_ERROR(
      CheckSameShape(k_cache, v_cache, "k_cache", "v_cache"));

  const int64_t tokens = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t block_size = k_cache.Dim(1);
  const int64_t kv_heads = k_cache.Dim(2);
  const int64_t max_blocks = block_table.Dim(1);

  if (k_cache.Dim(3) != head_dim) {
    return InvalidArgumentError("q head_dim is ", head_dim,
                                " but the cache's "
                                "is ",
                                k_cache.Dim(3));
  }
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    return InvalidArgumentError("q_heads (", q_heads,
                                ") is not a multiple of "
                                "kv_heads (",
                                kv_heads, ")");
  }
  if (seq_of_token.Dim(0) != tokens || q_pos.Dim(0) != tokens) {
    return InvalidArgumentError("seq_of_token/q_pos have ", seq_of_token.Dim(0),
                                "/", q_pos.Dim(0), " entries but there are ",
                                tokens, " tokens");
  }

  if (tokens == 0) return OkStatus();

  // The table's width is a scheduler configuration, not a property of this
  // batch. Sizing the tile from it made shared-memory demand a function of
  // max_seq_len rather than of the prompt.
  const int64_t table_keys = max_blocks * block_size;
  const int64_t max_keys =
      max_context > 0 ? std::min(max_context, table_keys) : table_keys;
  const int block = BlockFor(head_dim);

  const size_t smem =
      (static_cast<size_t>(block) + static_cast<size_t>(max_keys)) *
      sizeof(float);

  // Same 48 KB ceiling as the contiguous kernel, and the same reason it is not
  // a problem yet: this is the correctness path, and the tiled kernel that
  // lifts the limit is the one FlashInfer provides.
  if (smem > 48u * 1024u) {
    return ResourceExhaustedError("paged attention needs ", smem,
                                  " B of shared memory for ", max_keys,
                                  " keys, over the 48 KB limit");
  }

  PagedAttentionKernel<<<dim3(static_cast<unsigned>(tokens),
                              static_cast<unsigned>(q_heads)),
                         block, smem, stream>>>(
      static_cast<const bf16*>(q.Data()),
      static_cast<const bf16*>(k_cache.Data()),
      static_cast<const bf16*>(v_cache.Data()),
      static_cast<const int32_t*>(block_table.Data()),
      static_cast<const int32_t*>(seq_of_token.Data()),
      static_cast<const int32_t*>(q_pos.Data()), static_cast<bf16*>(out.Data()),
      q_heads, kv_heads, head_dim, block_size, max_blocks, scale, max_keys);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status PagedAttentionWithLse(const TensorView& q, const TensorView& k_cache,
                             const TensorView& v_cache,
                             const TensorView& block_table,
                             const TensorView& seq_of_token,
                             const TensorView& q_pos, const TensorView& out,
                             const TensorView& lse, float scale, int64_t window,
                             int64_t max_context, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(q, DataType::kBFloat16, 3, "q"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(k_cache, DataType::kBFloat16, 4, "k_cache"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(v_cache, DataType::kBFloat16, 4, "v_cache"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(block_table, DataType::kInt32, 2, "block_table"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(seq_of_token, DataType::kInt32, 1, "seq_of_token"));
  INFERX_RETURN_IF_ERROR(CheckTensor(q_pos, DataType::kInt32, 1, "q_pos"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(q, out, "q", "out"));
  INFERX_RETURN_IF_ERROR(
      CheckSameShape(k_cache, v_cache, "k_cache", "v_cache"));

  // lse is optional: an empty TensorView means "do not write it", which keeps a
  // caller that only wants the window from allocating a scratch it never reads.
  float* lse_ptr = nullptr;
  if (lse.IsDefined()) {
    INFERX_RETURN_IF_ERROR(CheckTensor(lse, DataType::kFloat, 2, "lse"));
    if (lse.Dim(0) != q.Dim(0) || lse.Dim(1) != q.Dim(1)) {
      return InvalidArgumentError("lse is [", lse.Dim(0), ", ", lse.Dim(1),
                                  "] but expected [", q.Dim(0), ", ", q.Dim(1),
                                  "]");
    }
    lse_ptr = static_cast<float*>(lse.Data());
  }

  const int64_t tokens = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t block_size = k_cache.Dim(1);
  const int64_t kv_heads = k_cache.Dim(2);
  const int64_t max_blocks = block_table.Dim(1);

  if (k_cache.Dim(3) != head_dim) {
    return InvalidArgumentError("q head_dim is ", head_dim,
                                " but the cache's "
                                "is ",
                                k_cache.Dim(3));
  }
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    return InvalidArgumentError("q_heads (", q_heads,
                                ") is not a multiple of "
                                "kv_heads (",
                                kv_heads, ")");
  }
  if (seq_of_token.Dim(0) != tokens || q_pos.Dim(0) != tokens) {
    return InvalidArgumentError("seq_of_token/q_pos have ", seq_of_token.Dim(0),
                                "/", q_pos.Dim(0), " entries but there are ",
                                tokens, " tokens");
  }

  if (tokens == 0) return OkStatus();

  // The tile holds one float per visible key. On a sliding layer the visible
  // key count is bounded by `window`, so sizing `max_context` from it (which
  // the caller does) keeps the shared-memory demand at `window` floats rather
  // than the full context.
  const int64_t table_keys = max_blocks * block_size;
  const int64_t max_keys =
      max_context > 0 ? std::min(max_context, table_keys) : table_keys;
  const int block = BlockFor(head_dim);

  const size_t smem =
      (static_cast<size_t>(block) + static_cast<size_t>(max_keys)) *
      sizeof(float);

  if (smem > 48u * 1024u) {
    return ResourceExhaustedError("paged attention (lse) needs ", smem,
                                  " B of shared memory for ", max_keys,
                                  " keys, over the 48 KB limit");
  }

  PagedAttentionWithLseKernel<<<dim3(static_cast<unsigned>(tokens),
                                     static_cast<unsigned>(q_heads)),
                                block, smem, stream>>>(
      static_cast<const bf16*>(q.Data()),
      static_cast<const bf16*>(k_cache.Data()),
      static_cast<const bf16*>(v_cache.Data()),
      static_cast<const int32_t*>(block_table.Data()),
      static_cast<const int32_t*>(seq_of_token.Data()),
      static_cast<const int32_t*>(q_pos.Data()), static_cast<bf16*>(out.Data()),
      lse_ptr, q_heads, kv_heads, head_dim, block_size, max_blocks, scale,
      window, max_keys);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

// Temperature + nucleus sampling for one row, in one block.
//
// Four block-wide passes over the row and no sort: max, then the partition
// function, then a bisection on the probability cutoff that defines the
// nucleus, then an inverse-CDF draw restricted to it. Each pass is a reduction,
// so the cost is linear in the vocabulary rather than n log n.
__global__ void SampleKernel(const bf16* __restrict__ logits,
                             const int32_t* __restrict__ rows,
                             const float* __restrict__ temperature,
                             const float* __restrict__ top_p,
                             const int32_t* __restrict__ top_k,
                             const float* __restrict__ min_p,
                             const uint64_t* __restrict__ seeds,
                             int32_t* __restrict__ out, int64_t vocab) {
  __shared__ float shared[256];
  __shared__ int shared_idx[256];
  __shared__ float s_cut;
  __shared__ float s_mass;

  const int r = static_cast<int>(blockIdx.x);
  const bf16* row = logits + static_cast<int64_t>(rows[r]) * vocab;

  const float t = temperature[r];
  const float p = top_p[r];
  const int k = top_k[r];
  const float mp = min_p[r];

  // Greedy is the temperature -> 0 limit, and the path most callers take.
  if (!(t > 0.0f)) {
    float best = -INFINITY;
    int best_i = 0;

    for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
      const float v = __bfloat162float(row[i]);
      if (v > best) {
        best = v;
        best_i = static_cast<int>(i);
      }
    }

    shared[threadIdx.x] = best;
    shared_idx[threadIdx.x] = best_i;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride &&
          shared[threadIdx.x + stride] > shared[threadIdx.x]) {
        shared[threadIdx.x] = shared[threadIdx.x + stride];
        shared_idx[threadIdx.x] = shared_idx[threadIdx.x + stride];
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) out[r] = shared_idx[0];
    return;
  }

  const float inv_t = 1.0f / t;

  // Pass 1: the maximum, so the exponentials below cannot overflow.
  float local_max = -INFINITY;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    local_max = fmaxf(local_max, __bfloat162float(row[i]) * inv_t);
  }

  shared[threadIdx.x] = local_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  const float max_logit = shared[0];
  __syncthreads();

  // Pass 2: the partition function.
  float local_sum = 0.0f;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    local_sum += __expf(__bfloat162float(row[i]) * inv_t - max_logit);
  }

  shared[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }

  const float total = shared[0];
  __syncthreads();

  // Pass 3: bisect the probability cutoff. The nucleus is every token whose
  // normalized probability is at least `cut`, where `cut` is the largest
  // threshold whose surviving mass still reaches p. Sorting would give the same
  // set; this gets there with reductions instead. top_k gets the same
  // treatment with a count in place of the mass, min_p is a closed-form
  // threshold (the max probability is exactly 1/total), and the final cut is
  // the strictest of the three.
  if (threadIdx.x == 0) {
    s_cut = 0.0f;
    s_mass = 1.0f;
  }
  __syncthreads();

  if (p < 1.0f) {
    float lo = 0.0f;
    float hi = 1.0f;

    for (int step = 0; step < 32; ++step) {
      const float mid = 0.5f * (lo + hi);

      float mass = 0.0f;
      for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float prob =
            __expf(__bfloat162float(row[i]) * inv_t - max_logit) / total;
        if (prob >= mid) mass += prob;
      }

      shared[threadIdx.x] = mass;
      __syncthreads();

      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
      }

      // Raise the floor while the nucleus is still big enough to hold p.
      if (shared[0] >= p) {
        lo = mid;
        if (threadIdx.x == 0) {
          s_cut = mid;
          s_mass = shared[0];
        }
      } else {
        hi = mid;
      }
      __syncthreads();
    }
  }

  float cut = s_cut;

  // top_k: the largest cut that still keeps at least k tokens. Ties at the
  // k-th probability keep more than k, which is the conventional resolution.
  if (k > 0 && static_cast<int64_t>(k) < vocab) {
    __shared__ float s_cut_k;
    if (threadIdx.x == 0) s_cut_k = 0.0f;
    __syncthreads();

    float lo = 0.0f;
    float hi = 1.0f;
    for (int step = 0; step < 32; ++step) {
      const float mid = 0.5f * (lo + hi);

      float count = 0.0f;
      for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
        const float prob =
            __expf(__bfloat162float(row[i]) * inv_t - max_logit) / total;
        if (prob >= mid) count += 1.0f;
      }

      shared[threadIdx.x] = count;
      __syncthreads();
      for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
          shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
      }

      if (shared[0] >= static_cast<float>(k)) {
        lo = mid;
        if (threadIdx.x == 0) s_cut_k = mid;
      } else {
        hi = mid;
      }
      __syncthreads();
    }
    cut = fmaxf(cut, s_cut_k);
    __syncthreads();
  }

  // min_p: relative to the most probable token, whose normalized probability
  // is exp(0)/total.
  if (mp > 0.0f) cut = fmaxf(cut, mp / total);

  float mass = s_mass;

  // The surviving mass is only known for the top_p cut; if top_k or min_p
  // tightened it, one more reduction re-measures the survivors so the draw
  // below normalizes correctly.
  if (cut > s_cut) {
    float local = 0.0f;
    for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
      const float prob =
          __expf(__bfloat162float(row[i]) * inv_t - max_logit) / total;
      if (prob >= cut) local += prob;
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride)
        shared[threadIdx.x] += shared[threadIdx.x + stride];
      __syncthreads();
    }
    mass = shared[0];
    __syncthreads();
  }

  // Pass 4: inverse CDF over the nucleus. Philox rather than a counter hashed
  // by hand, so the stream is well-distributed and reproducible from the seed.
  curandStatePhilox4_32_10_t rng;
  curand_init(seeds[r], 0, 0, &rng);
  const float target = curand_uniform(&rng) * mass;

  // Scanned by a single thread. The nucleus is small -- top_p of 0.9 is
  // typically tens of tokens -- and a parallel scan over 151936 entries to
  // find one crossing point would cost more than it saves.
  if (threadIdx.x == 0) {
    float running = 0.0f;
    int chosen = -1;

    for (int64_t i = 0; i < vocab; ++i) {
      const float prob =
          __expf(__bfloat162float(row[i]) * inv_t - max_logit) / total;

      if (prob < cut) continue;

      running += prob;
      if (running >= target) {
        chosen = static_cast<int>(i);
        break;
      }
    }

    // Rounding can leave `target` a hair above the accumulated mass; the last
    // nucleus member is the right answer there, not a failure.
    if (chosen < 0) {
      for (int64_t i = vocab - 1; i >= 0; --i) {
        const float prob =
            __expf(__bfloat162float(row[i]) * inv_t - max_logit) / total;
        if (prob >= cut) {
          chosen = static_cast<int>(i);
          break;
        }
      }
    }

    out[r] = chosen < 0 ? 0 : chosen;
  }
}

// Penalties and min-tokens stop masking, in place, before sampling. One block
// per row; the history holds *unique* ids (the host aggregates counts), so
// every entry touches a distinct logit and no atomics are needed. Rows with
// nothing to do read three floats and exit, which is what makes it safe to
// leave in the captured graph unconditionally.
__global__ void ApplyPenaltiesKernel(
    bf16* __restrict__ logits, const int32_t* __restrict__ rows,
    const float* __restrict__ presence, const float* __restrict__ frequency,
    const float* __restrict__ repetition,
    const int32_t* __restrict__ history_ids,
    const int32_t* __restrict__ history_counts, int hist_cap,
    const int32_t* __restrict__ mask_ids, int mask_cap, int64_t vocab) {
  const int r = static_cast<int>(blockIdx.x);
  bf16* row = logits + static_cast<int64_t>(rows[r]) * vocab;

  const float pres = presence[r];
  const float freq = frequency[r];
  const float rep = repetition[r];

  if (pres != 0.0f || freq != 0.0f || rep != 1.0f) {
    for (int i = static_cast<int>(threadIdx.x); i < hist_cap;
         i += static_cast<int>(blockDim.x)) {
      const int32_t id = history_ids[r * hist_cap + i];
      if (id < 0 || static_cast<int64_t>(id) >= vocab) continue;

      float value = __bfloat162float(row[id]);
      if (rep != 1.0f) value = value > 0.0f ? value / rep : value * rep;
      value -= pres;
      value -= freq * static_cast<float>(history_counts[r * hist_cap + i]);
      row[id] = __float2bfloat16(value);
    }
  }

  for (int i = static_cast<int>(threadIdx.x); i < mask_cap;
       i += static_cast<int>(blockDim.x)) {
    const int32_t id = mask_ids[r * mask_cap + i];
    if (id >= 0 && static_cast<int64_t>(id) < vocab) {
      row[id] = __float2bfloat16(-INFINITY);
    }
  }
}

// Logprobs of the sampled token and its top-k alternatives, over the
// (post-penalty) logits at temperature 1. Two reductions for the partition
// function, then k rounds of "argmax strictly after the previous winner" in
// (value, index) order -- k is at most 20, so k more passes beat any sort.
__global__ void ComputeLogprobsKernel(
    const bf16* __restrict__ logits, const int32_t* __restrict__ rows,
    const int32_t* __restrict__ chosen, const int32_t* __restrict__ k_wanted,
    float* __restrict__ chosen_lp, int32_t* __restrict__ top_ids,
    float* __restrict__ top_lps, int max_k, int64_t vocab) {
  __shared__ float shared[256];
  __shared__ int shared_idx[256];
  __shared__ float s_prev_val;
  __shared__ int s_prev_idx;

  const int r = static_cast<int>(blockIdx.x);
  const int k = min(k_wanted[r], max_k);
  if (k < 0) return;

  const bf16* row = logits + static_cast<int64_t>(rows[r]) * vocab;

  // Partition function at temperature 1.
  float local_max = -INFINITY;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    local_max = fmaxf(local_max, __bfloat162float(row[i]));
  }
  shared[threadIdx.x] = local_max;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared[threadIdx.x] =
          fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
    }
    __syncthreads();
  }
  const float max_logit = shared[0];
  __syncthreads();

  float local_sum = 0.0f;
  for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
    local_sum += __expf(__bfloat162float(row[i]) - max_logit);
  }
  shared[threadIdx.x] = local_sum;
  __syncthreads();
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride)
      shared[threadIdx.x] += shared[threadIdx.x + stride];
    __syncthreads();
  }
  const float log_z = __logf(shared[0]) + max_logit;
  __syncthreads();

  if (threadIdx.x == 0) {
    chosen_lp[r] = __bfloat162float(row[chosen[r]]) - log_z;
    s_prev_val = INFINITY;
    s_prev_idx = -1;
  }
  __syncthreads();

  for (int j = 0; j < k; ++j) {
    const float prev_val = s_prev_val;
    const int prev_idx = s_prev_idx;

    // The next winner is the largest element strictly after the previous one
    // in descending (value, ascending index) order, so equal values resolve
    // deterministically.
    float best = -INFINITY;
    int best_i = -1;
    for (int64_t i = threadIdx.x; i < vocab; i += blockDim.x) {
      const float v = __bfloat162float(row[i]);
      const bool after_prev =
          v < prev_val || (v == prev_val && static_cast<int>(i) > prev_idx);
      if (!after_prev) continue;
      if (v > best || (v == best && static_cast<int>(i) < best_i)) {
        best = v;
        best_i = static_cast<int>(i);
      }
    }
    shared[threadIdx.x] = best;
    shared_idx[threadIdx.x] = best_i;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (threadIdx.x < stride) {
        const float other = shared[threadIdx.x + stride];
        const int other_i = shared_idx[threadIdx.x + stride];
        if (other > shared[threadIdx.x] ||
            (other == shared[threadIdx.x] && other_i >= 0 &&
             (shared_idx[threadIdx.x] < 0 ||
              other_i < shared_idx[threadIdx.x]))) {
          shared[threadIdx.x] = other;
          shared_idx[threadIdx.x] = other_i;
        }
      }
      __syncthreads();
    }

    if (threadIdx.x == 0) {
      const int idx = shared_idx[0];
      top_ids[r * max_k + j] = idx;
      top_lps[r * max_k + j] = idx < 0 ? -INFINITY : shared[0] - log_z;
      s_prev_val = shared[0];
      s_prev_idx = idx;
    }
    __syncthreads();
  }

  // Pad the tail so a reader never confuses stale entries for results.
  for (int j = k + static_cast<int>(threadIdx.x); j < max_k;
       j += static_cast<int>(blockDim.x)) {
    top_ids[r * max_k + j] = -1;
    top_lps[r * max_k + j] = -INFINITY;
  }
}

Status ArgmaxSample(const TensorView& logits, const TensorView& rows,
                    const TensorView& out, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(logits, DataType::kBFloat16, 2, "logits"));
  INFERX_RETURN_IF_ERROR(CheckTensor(rows, DataType::kInt32, 1, "rows"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kInt32, 1, "out"));

  const int64_t n = rows.Dim(0);
  const int64_t vocab = logits.Dim(1);

  if (out.Dim(0) != n) {
    return InvalidArgumentError("out has ", out.Dim(0), " entries but ", n,
                                " rows were requested");
  }

  if (n == 0) return OkStatus();

  ArgmaxKernel<<<static_cast<unsigned>(n), 256, 0, stream>>>(
      static_cast<const bf16*>(logits.Data()),
      static_cast<const int32_t*>(rows.Data()),
      static_cast<int32_t*>(out.Data()), vocab);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status SampleTokens(const TensorView& logits, const TensorView& rows,
                    const TensorView& temperature, const TensorView& top_p,
                    const TensorView& top_k, const TensorView& min_p,
                    const TensorView& seeds, const TensorView& out,
                    Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(logits, DataType::kBFloat16, 2, "logits"));
  INFERX_RETURN_IF_ERROR(CheckTensor(rows, DataType::kInt32, 1, "rows"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(temperature, DataType::kFloat, 1, "temperature"));
  INFERX_RETURN_IF_ERROR(CheckTensor(top_p, DataType::kFloat, 1, "top_p"));
  INFERX_RETURN_IF_ERROR(CheckTensor(top_k, DataType::kInt32, 1, "top_k"));
  INFERX_RETURN_IF_ERROR(CheckTensor(min_p, DataType::kFloat, 1, "min_p"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kInt32, 1, "out"));

  const int64_t n = rows.Dim(0);
  const int64_t vocab = logits.Dim(1);

  for (const auto& [t, name] :
       {std::pair{temperature, "temperature"}, std::pair{top_p, "top_p"},
        std::pair{min_p, "min_p"}}) {
    if (t.Dim(0) != n) {
      return InvalidArgumentError(name, " has ", t.Dim(0), " entries but ", n,
                                  " rows were requested");
    }
  }

  if (top_k.Dim(0) != n) {
    return InvalidArgumentError("top_k has ", top_k.Dim(0), " entries but ", n,
                                " rows were requested");
  }

  if (seeds.Dim(0) != n) {
    return InvalidArgumentError("seeds has ", seeds.Dim(0), " entries but ", n,
                                " rows were requested");
  }

  if (out.Dim(0) != n) {
    return InvalidArgumentError("out has ", out.Dim(0), " entries but ", n,
                                " rows were requested");
  }

  if (n == 0) return OkStatus();

  SampleKernel<<<static_cast<unsigned>(n), 256, 0, stream>>>(
      static_cast<const bf16*>(logits.Data()),
      static_cast<const int32_t*>(rows.Data()),
      static_cast<const float*>(temperature.Data()),
      static_cast<const float*>(top_p.Data()),
      static_cast<const int32_t*>(top_k.Data()),
      static_cast<const float*>(min_p.Data()),
      static_cast<const uint64_t*>(seeds.Data()),
      static_cast<int32_t*>(out.Data()), vocab);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status ApplyPenalties(const TensorView& logits, const TensorView& rows,
                      const TensorView& presence, const TensorView& frequency,
                      const TensorView& repetition,
                      const TensorView& history_ids,
                      const TensorView& history_counts,
                      const TensorView& mask_ids, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(logits, DataType::kBFloat16, 2, "logits"));
  INFERX_RETURN_IF_ERROR(CheckTensor(rows, DataType::kInt32, 1, "rows"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(presence, DataType::kFloat, 1, "presence"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(frequency, DataType::kFloat, 1, "frequency"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(repetition, DataType::kFloat, 1, "repetition"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(history_ids, DataType::kInt32, 2, "history_ids"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(history_counts, DataType::kInt32, 2, "history_counts"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(mask_ids, DataType::kInt32, 2, "mask_ids"));

  const int64_t n = rows.Dim(0);
  const int64_t vocab = logits.Dim(1);
  const int64_t hist_cap = history_ids.Dim(1);
  const int64_t mask_cap = mask_ids.Dim(1);

  for (const auto& [t, name] :
       {std::pair{presence, "presence"}, std::pair{frequency, "frequency"},
        std::pair{repetition, "repetition"}}) {
    if (t.Dim(0) != n) {
      return InvalidArgumentError(name, " has ", t.Dim(0), " entries but ", n,
                                  " rows were requested");
    }
  }
  if (history_ids.Dim(0) != n || history_counts.Dim(0) != n ||
      mask_ids.Dim(0) != n || history_counts.Dim(1) != hist_cap) {
    return InvalidArgumentError(
        "penalty history/mask shapes disagree with the row count");
  }

  if (n == 0) return OkStatus();

  ApplyPenaltiesKernel<<<static_cast<unsigned>(n), 128, 0, stream>>>(
      static_cast<bf16*>(logits.Data()),
      static_cast<const int32_t*>(rows.Data()),
      static_cast<const float*>(presence.Data()),
      static_cast<const float*>(frequency.Data()),
      static_cast<const float*>(repetition.Data()),
      static_cast<const int32_t*>(history_ids.Data()),
      static_cast<const int32_t*>(history_counts.Data()),
      static_cast<int>(hist_cap), static_cast<const int32_t*>(mask_ids.Data()),
      static_cast<int>(mask_cap), vocab);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status ComputeLogprobs(const TensorView& logits, const TensorView& rows,
                       const TensorView& chosen, const TensorView& k_wanted,
                       const TensorView& chosen_lp, const TensorView& top_ids,
                       const TensorView& top_lps, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(logits, DataType::kBFloat16, 2, "logits"));
  INFERX_RETURN_IF_ERROR(CheckTensor(rows, DataType::kInt32, 1, "rows"));
  INFERX_RETURN_IF_ERROR(CheckTensor(chosen, DataType::kInt32, 1, "chosen"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(k_wanted, DataType::kInt32, 1, "k_wanted"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(chosen_lp, DataType::kFloat, 1, "chosen_lp"));
  INFERX_RETURN_IF_ERROR(CheckTensor(top_ids, DataType::kInt32, 2, "top_ids"));
  INFERX_RETURN_IF_ERROR(CheckTensor(top_lps, DataType::kFloat, 2, "top_lps"));

  const int64_t n = rows.Dim(0);
  const int64_t vocab = logits.Dim(1);
  const int64_t max_k = top_ids.Dim(1);

  if (chosen.Dim(0) != n || k_wanted.Dim(0) != n || chosen_lp.Dim(0) != n ||
      top_ids.Dim(0) != n || top_lps.Dim(0) != n || top_lps.Dim(1) != max_k) {
    return InvalidArgumentError(
        "logprob output shapes disagree with the row "
        "count");
  }

  if (n == 0) return OkStatus();

  ComputeLogprobsKernel<<<static_cast<unsigned>(n), 256, 0, stream>>>(
      static_cast<const bf16*>(logits.Data()),
      static_cast<const int32_t*>(rows.Data()),
      static_cast<const int32_t*>(chosen.Data()),
      static_cast<const int32_t*>(k_wanted.Data()),
      static_cast<float*>(chosen_lp.Data()),
      static_cast<int32_t*>(top_ids.Data()),
      static_cast<float*>(top_lps.Data()), static_cast<int>(max_k), vocab);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status ScatterTokens(const TensorView& src, const TensorView& dst,
                     const TensorView& slots, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(src, DataType::kInt32, 1, "src"));
  INFERX_RETURN_IF_ERROR(CheckTensor(dst, DataType::kInt32, 1, "dst"));
  INFERX_RETURN_IF_ERROR(CheckTensor(slots, DataType::kInt32, 1, "slots"));

  const int64_t n = src.Dim(0);

  if (slots.Dim(0) != n) {
    return InvalidArgumentError("slots has ", slots.Dim(0),
                                " entries but src "
                                "has ",
                                n);
  }

  if (n == 0) return OkStatus();

  constexpr int kBlock = 128;
  const unsigned grid = static_cast<unsigned>((n + kBlock - 1) / kBlock);

  ScatterTokensKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const int32_t*>(src.Data()),
      static_cast<int32_t*>(dst.Data()),
      static_cast<const int32_t*>(slots.Data()), n, dst.Dim(0));

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status EmbeddingLookup(const TensorView& table, const TensorView& ids,
                       const TensorView& out, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(table, DataType::kBFloat16, 2, "table"));
  INFERX_RETURN_IF_ERROR(CheckTensor(ids, DataType::kInt32, 1, "ids"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));

  const int64_t tokens = ids.Dim(0);
  const int64_t hidden = table.Dim(1);
  const int64_t vocab = table.Dim(0);

  if (out.Dim(0) != tokens || out.Dim(1) != hidden) {
    return InvalidArgumentError("out is ", out.GetShape().ToString(),
                                ", expected [", tokens, ", ", hidden, "]");
  }

  if (tokens == 0) return OkStatus();

  const int block = BlockFor(hidden);

  EmbeddingKernel<<<static_cast<unsigned>(tokens), block, 0, stream>>>(
      static_cast<const bf16*>(table.Data()),
      static_cast<const int32_t*>(ids.Data()), static_cast<bf16*>(out.Data()),
      hidden, vocab);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status SplitQkvWithBias(const TensorView& fused, const TensorView& bias,
                        const TensorView& q, const TensorView& k,
                        const TensorView& v, Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(fused, DataType::kBFloat16, 2, "fused"));
  INFERX_RETURN_IF_ERROR(CheckTensor(q, DataType::kBFloat16, 2, "q"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k, DataType::kBFloat16, 2, "k"));
  INFERX_RETURN_IF_ERROR(CheckTensor(v, DataType::kBFloat16, 2, "v"));

  const int64_t tokens = fused.Dim(0);
  const int64_t q_dim = q.Dim(1);
  const int64_t kv_dim = k.Dim(1);

  if (fused.Dim(1) != q_dim + 2 * kv_dim) {
    return InvalidArgumentError("fused is ", fused.Dim(1), " wide but q_dim ",
                                q_dim, " + 2*kv_dim ", kv_dim, " is ",
                                q_dim + 2 * kv_dim);
  }
  if (q.Dim(0) != tokens || k.Dim(0) != tokens || v.Dim(0) != tokens) {
    return InvalidArgumentError("q/k/v token counts disagree with fused");
  }
  if (v.Dim(1) != kv_dim) {
    return InvalidArgumentError("k is ", kv_dim, " wide but v is ", v.Dim(1));
  }

  const bf16* bias_ptr = nullptr;
  if (bias.IsDefined()) {
    INFERX_RETURN_IF_ERROR(CheckTensor(bias, DataType::kBFloat16, 1, "bias"));
    if (bias.Dim(0) != fused.Dim(1)) {
      return InvalidArgumentError("bias has ", bias.Dim(0),
                                  " entries but "
                                  "fused is ",
                                  fused.Dim(1), " wide");
    }
    bias_ptr = static_cast<const bf16*>(bias.Data());
  }

  if (tokens == 0) return OkStatus();

  constexpr int kBlock = 256;
  const int64_t total = tokens * fused.Dim(1);
  const int64_t want = (total + kBlock - 1) / kBlock;
  const unsigned grid = static_cast<unsigned>(want > 4096 ? 4096 : want);

  SplitQkvKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const bf16*>(fused.Data()), bias_ptr,
      static_cast<bf16*>(q.Data()), static_cast<bf16*>(k.Data()),
      static_cast<bf16*>(v.Data()), tokens, q_dim, kv_dim);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status SiluMulFused(const TensorView& fused, const TensorView& out,
                    Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(fused, DataType::kBFloat16, 2, "fused"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));

  const int64_t tokens = out.Dim(0);
  const int64_t inter = out.Dim(1);

  if (fused.Dim(0) != tokens || fused.Dim(1) != 2 * inter) {
    return InvalidArgumentError("fused is ", fused.GetShape().ToString(),
                                ", expected [", tokens, ", ", 2 * inter, "]");
  }

  if (tokens == 0) return OkStatus();

  constexpr int kBlock = 256;
  const int64_t total = tokens * inter;
  const int64_t want = (total + kBlock - 1) / kBlock;
  const unsigned grid = static_cast<unsigned>(want > 4096 ? 4096 : want);

  SiluMulFusedKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const bf16*>(fused.Data()), static_cast<bf16*>(out.Data()),
      tokens, inter);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status AddBiasInPlace(const TensorView& out, const TensorView& bias,
                      Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));
  INFERX_RETURN_IF_ERROR(CheckTensor(bias, DataType::kBFloat16, 1, "bias"));

  const int64_t tokens = out.Dim(0);
  const int64_t width = out.Dim(1);

  if (bias.Dim(0) != width) {
    return InvalidArgumentError("bias has ", bias.Dim(0),
                                " elements but out "
                                "is ",
                                width, " wide");
  }

  if (tokens == 0) return OkStatus();

  constexpr int kBlock = 256;
  const int64_t grid_want = (tokens * width + kBlock - 1) / kBlock;
  const unsigned grid =
      static_cast<unsigned>(grid_want > 4096 ? 4096 : grid_want);

  AddBiasKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<bf16*>(out.Data()), static_cast<const bf16*>(bias.Data()),
      tokens, width);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status AddInPlace(const TensorView& out, const TensorView& residual,
                  Stream stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(residual, DataType::kBFloat16, 2, "residual"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(out, residual, "out", "residual"));

  const int64_t n = out.Numel();
  if (n == 0) return OkStatus();

  constexpr int kBlock = 256;
  const int64_t grid_want = (n + kBlock - 1) / kBlock;
  const unsigned grid =
      static_cast<unsigned>(grid_want > 4096 ? 4096 : grid_want);

  AddKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<bf16*>(out.Data()), static_cast<const bf16*>(residual.Data()),
      n);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace inferx::ops
