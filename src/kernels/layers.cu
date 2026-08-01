#include "inferx/kernels/layers.h"

#include <algorithm>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"

namespace inferx::kernels {
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
// target, but the loops are written as grid-strides so an odd width still works.
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
    const float inv_freq =
        __powf(theta, -2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
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
  float* tile = smem;              // blockDim.x floats for reductions
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

// One block per (query token, query head), same as the contiguous kernel. The
// only difference is where a key comes from: `block_table[seq][j / block_size]`
// gives the block, `j % block_size` the slot inside it.
__global__ void PagedAttentionKernel(
    const bf16* __restrict__ q, const bf16* __restrict__ k_cache,
    const bf16* __restrict__ v_cache, const int32_t* __restrict__ block_table,
    const int32_t* __restrict__ seq_of_token, const int32_t* __restrict__ q_pos,
    bf16* __restrict__ out, int64_t q_heads, int64_t kv_heads,
    int64_t head_dim, int64_t block_size, int64_t max_blocks, float scale,
    int64_t max_keys) {
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
      acc += scores[j] * ToF32(v_cache[(slot * kv_heads + kv_head) * head_dim + d]);
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

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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

  for (int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
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

Status CheckSameShape(const TensorView& a, const TensorView& b,
                      const char* an, const char* bn) {
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
               const TensorView& out, float eps, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(x, DataType::kBFloat16, 2, "x"));
  INFERX_RETURN_IF_ERROR(CheckTensor(weight, DataType::kBFloat16, 1, "weight"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(x, out, "x", "out"));

  const int64_t tokens = x.Dim(0);
  const int64_t hidden = x.Dim(1);

  if (weight.Dim(0) != hidden) {
    return InvalidArgumentError("weight has ", weight.Dim(0), " elements but "
                                "hidden is ", hidden);
  }

  if (tokens == 0) return OkStatus();

  const int block = BlockFor(hidden);

  RmsNormKernel<<<static_cast<unsigned>(tokens), block,
                  block * sizeof(float), stream>>>(
      static_cast<const bf16*>(x.Data()),
      static_cast<const bf16*>(weight.Data()),
      static_cast<bf16*>(out.Data()), hidden, eps);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status RotaryEmbedding(const TensorView& q, const TensorView& k,
                       const TensorView& positions, float theta,
                       cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(q, DataType::kBFloat16, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k, DataType::kBFloat16, 3, "k"));
  INFERX_RETURN_IF_ERROR(CheckTensor(positions, DataType::kInt32, 1,
                                     "positions"));

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
               const TensorView& out, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(gate, DataType::kBFloat16, 2, "gate"));
  INFERX_RETURN_IF_ERROR(CheckTensor(up, DataType::kBFloat16, 2, "up"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(gate, up, "gate", "up"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(gate, out, "gate", "out"));

  const int64_t n = gate.Numel();
  if (n == 0) return OkStatus();

  constexpr int kBlock = 256;
  const int64_t grid_want = (n + kBlock - 1) / kBlock;
  const unsigned grid = static_cast<unsigned>(grid_want > 4096 ? 4096
                                                               : grid_want);

  SiluMulKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const bf16*>(gate.Data()),
      static_cast<const bf16*>(up.Data()),
      static_cast<bf16*>(out.Data()), n);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status Attention(const TensorView& q, const TensorView& k, const TensorView& v,
                 const TensorView& out, float scale, cudaStream_t stream) {
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
    return InvalidArgumentError("q_heads (", q_heads, ") is not a multiple of "
                                "kv_heads (", kv_heads, ")");
  }

  if (tokens == 0) return OkStatus();

  const int block = BlockFor(head_dim);

  // Shared memory holds the reduction tile plus one score per key. The score
  // array is sized for the longest query, which is the last token.
  const size_t smem = (static_cast<size_t>(block) +
                       static_cast<size_t>(tokens)) * sizeof(float);

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
                       const TensorView& slots, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(k, DataType::kBFloat16, 3, "k"));
  INFERX_RETURN_IF_ERROR(CheckTensor(v, DataType::kBFloat16, 3, "v"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k_cache, DataType::kBFloat16, 4,
                                     "k_cache"));
  INFERX_RETURN_IF_ERROR(CheckTensor(v_cache, DataType::kBFloat16, 4,
                                     "v_cache"));
  INFERX_RETURN_IF_ERROR(CheckTensor(slots, DataType::kInt32, 1, "slots"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(k, v, "k", "v"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(k_cache, v_cache, "k_cache",
                                        "v_cache"));

  const int64_t tokens = k.Dim(0);
  const int64_t kv_heads = k.Dim(1);
  const int64_t head_dim = k.Dim(2);

  if (slots.Dim(0) != tokens) {
    return InvalidArgumentError("slots has ", slots.Dim(0), " entries but "
                                "there are ", tokens, " tokens");
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

Status PagedAttention(const TensorView& q, const TensorView& k_cache,
                      const TensorView& v_cache, const TensorView& block_table,
                      const TensorView& seq_of_token, const TensorView& q_pos,
                      const TensorView& out, float scale, int64_t max_context,
                      cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(q, DataType::kBFloat16, 3, "q"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k_cache, DataType::kBFloat16, 4,
                                     "k_cache"));
  INFERX_RETURN_IF_ERROR(CheckTensor(v_cache, DataType::kBFloat16, 4,
                                     "v_cache"));
  INFERX_RETURN_IF_ERROR(CheckTensor(block_table, DataType::kInt32, 2,
                                     "block_table"));
  INFERX_RETURN_IF_ERROR(CheckTensor(seq_of_token, DataType::kInt32, 1,
                                     "seq_of_token"));
  INFERX_RETURN_IF_ERROR(CheckTensor(q_pos, DataType::kInt32, 1, "q_pos"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 3, "out"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(q, out, "q", "out"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(k_cache, v_cache, "k_cache",
                                        "v_cache"));

  const int64_t tokens = q.Dim(0);
  const int64_t q_heads = q.Dim(1);
  const int64_t head_dim = q.Dim(2);
  const int64_t block_size = k_cache.Dim(1);
  const int64_t kv_heads = k_cache.Dim(2);
  const int64_t max_blocks = block_table.Dim(1);

  if (k_cache.Dim(3) != head_dim) {
    return InvalidArgumentError("q head_dim is ", head_dim, " but the cache's "
                                "is ", k_cache.Dim(3));
  }
  if (kv_heads == 0 || q_heads % kv_heads != 0) {
    return InvalidArgumentError("q_heads (", q_heads, ") is not a multiple of "
                                "kv_heads (", kv_heads, ")");
  }
  if (seq_of_token.Dim(0) != tokens || q_pos.Dim(0) != tokens) {
    return InvalidArgumentError("seq_of_token/q_pos have ",
                                seq_of_token.Dim(0), "/", q_pos.Dim(0),
                                " entries but there are ", tokens, " tokens");
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
    return ResourceExhaustedError(
        "paged attention needs ", smem, " B of shared memory for ", max_keys,
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
      static_cast<const int32_t*>(q_pos.Data()),
      static_cast<bf16*>(out.Data()), q_heads, kv_heads, head_dim, block_size,
      max_blocks, scale, max_keys);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status ArgmaxSample(const TensorView& logits, const TensorView& rows,
                    const TensorView& out, cudaStream_t stream) {
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

Status ScatterTokens(const TensorView& src, const TensorView& dst,
                     const TensorView& slots, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(src, DataType::kInt32, 1, "src"));
  INFERX_RETURN_IF_ERROR(CheckTensor(dst, DataType::kInt32, 1, "dst"));
  INFERX_RETURN_IF_ERROR(CheckTensor(slots, DataType::kInt32, 1, "slots"));

  const int64_t n = src.Dim(0);

  if (slots.Dim(0) != n) {
    return InvalidArgumentError("slots has ", slots.Dim(0), " entries but src "
                                "has ", n);
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
                       const TensorView& out, cudaStream_t stream) {
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
      static_cast<const int32_t*>(ids.Data()),
      static_cast<bf16*>(out.Data()), hidden, vocab);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status SplitQkvWithBias(const TensorView& fused, const TensorView& bias,
                        const TensorView& q, const TensorView& k,
                        const TensorView& v, cudaStream_t stream) {
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
      return InvalidArgumentError("bias has ", bias.Dim(0), " entries but "
                                  "fused is ", fused.Dim(1), " wide");
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
                    cudaStream_t stream) {
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
                      cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));
  INFERX_RETURN_IF_ERROR(CheckTensor(bias, DataType::kBFloat16, 1, "bias"));

  const int64_t tokens = out.Dim(0);
  const int64_t width = out.Dim(1);

  if (bias.Dim(0) != width) {
    return InvalidArgumentError("bias has ", bias.Dim(0), " elements but out "
                                "is ", width, " wide");
  }

  if (tokens == 0) return OkStatus();

  constexpr int kBlock = 256;
  const int64_t grid_want = (tokens * width + kBlock - 1) / kBlock;
  const unsigned grid = static_cast<unsigned>(grid_want > 4096 ? 4096
                                                               : grid_want);

  AddBiasKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<bf16*>(out.Data()), static_cast<const bf16*>(bias.Data()),
      tokens, width);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status AddInPlace(const TensorView& out, const TensorView& residual,
                  cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));
  INFERX_RETURN_IF_ERROR(CheckTensor(residual, DataType::kBFloat16, 2,
                                     "residual"));
  INFERX_RETURN_IF_ERROR(CheckSameShape(out, residual, "out", "residual"));

  const int64_t n = out.Numel();
  if (n == 0) return OkStatus();

  constexpr int kBlock = 256;
  const int64_t grid_want = (n + kBlock - 1) / kBlock;
  const unsigned grid = static_cast<unsigned>(grid_want > 4096 ? 4096
                                                               : grid_want);

  AddKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<bf16*>(out.Data()),
      static_cast<const bf16*>(residual.Data()), n);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace inferx::kernels
