#include "inferx/kernels/mla.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kBlock = 256;

__device__ inline float ToF32(bf16 x) { return __bfloat162float(x); }
__device__ inline bf16 ToBf16(float x) { return __float2bfloat16(x); }

// The same rotation RopeKernel applies, over a sub-vector rather than a whole
// head. `inv_freq` is indexed by position within the *sub-vector* and scaled by
// its width, not the head's -- the rotated part has its own frequency ladder,
// which is what "decoupled" means here.
__global__ void RopeInPlaceKernel(bf16* __restrict__ x,
                                  const int32_t* __restrict__ positions,
                                  int64_t heads, int64_t head_dim,
                                  int64_t rope_dim, float theta) {
  const int64_t token = blockIdx.x;
  const int64_t head = blockIdx.y;
  const int64_t half = rope_dim / 2;
  const float pos = static_cast<float>(positions[token]);

  bf16* row = x + (token * heads + head) * head_dim + (head_dim - rope_dim);

  for (int64_t j = threadIdx.x; j < half; j += blockDim.x) {
    const float inv_freq = __powf(
        theta, -2.0f * static_cast<float>(j) / static_cast<float>(rope_dim));
    const float angle = pos * inv_freq;

    float s, c;
    __sincosf(angle, &s, &c);

    const float lo = ToF32(row[j]);
    const float hi = ToF32(row[j + half]);
    row[j] = ToBf16(lo * c - hi * s);
    row[j + half] = ToBf16(hi * c + lo * s);
  }
}

__global__ void SplitTrailingKernel(const bf16* __restrict__ src,
                                    bf16* __restrict__ head,
                                    bf16* __restrict__ tail, int64_t heads,
                                    int64_t head_width, int64_t tail_width) {
  const int64_t row = blockIdx.x;
  const int64_t h = blockIdx.y;
  const int64_t width = head_width + tail_width;

  const bf16* s = src + (row * heads + h) * width;

  for (int64_t c = threadIdx.x; c < width; c += blockDim.x) {
    if (c < head_width) {
      head[(row * heads + h) * head_width + c] = s[c];
    } else {
      tail[(row * heads + h) * tail_width + (c - head_width)] = s[c];
    }
  }
}

__global__ void AppendLatentKernel(const bf16* __restrict__ latent,
                                   const bf16* __restrict__ rope_key,
                                   bf16* __restrict__ cache,
                                   const int32_t* __restrict__ slot_mapping,
                                   int64_t latent_dim, int64_t rope_dim) {
  const int64_t token = blockIdx.x;
  const int64_t slot = slot_mapping[token];
  const int64_t width = latent_dim + rope_dim;

  bf16* dst = cache + slot * width;

  for (int64_t c = threadIdx.x; c < width; c += blockDim.x) {
    dst[c] = c < latent_dim ? latent[token * latent_dim + c]
                            : rope_key[token * rope_dim + (c - latent_dim)];
  }
}

__global__ void GatherLatentsKernel(const bf16* __restrict__ cache,
                                    const int32_t* __restrict__ block_table,
                                    bf16* __restrict__ out, int64_t width,
                                    int64_t block_size) {
  const int64_t token = blockIdx.x;
  const int64_t block = block_table[token / block_size];
  const int64_t slot = block * block_size + (token % block_size);

  const bf16* src = cache + slot * width;
  bf16* dst = out + token * width;

  for (int64_t c = threadIdx.x; c < width; c += blockDim.x) dst[c] = src[c];
}

// One block per (query token, head). Two passes over the context: the first
// finds the row maximum for a stable softmax, the second accumulates the
// weighted sum of V. Both are the same shape as the reference attention kernel
// in layers.cu, and for the same reason -- this is the correctness path, and a
// flash-style single pass belongs behind the same interface once there is a
// checkpoint to measure it against.
__global__ void MlaAttentionKernel(const bf16* __restrict__ q_nope,
                                   const bf16* __restrict__ q_rope,
                                   const bf16* __restrict__ k_nope,
                                   const bf16* __restrict__ k_rope,
                                   const bf16* __restrict__ v,
                                   bf16* __restrict__ out, int64_t heads,
                                   int64_t nope_dim, int64_t rope_dim,
                                   int64_t v_dim, int64_t context,
                                   int64_t query_base, float scale) {
  const int64_t query = blockIdx.x;
  const int64_t head = blockIdx.y;

  // Keys at positions after this query are not visible to it.
  const int64_t visible = query_base + query + 1;

  const bf16* qn = q_nope + (query * heads + head) * nope_dim;
  const bf16* qr = q_rope + (query * heads + head) * rope_dim;

  __shared__ float reduce[kBlock];

  // Pass 1: the maximum score.
  float local_max = -INFINITY;
  for (int64_t j = threadIdx.x; j < visible; j += blockDim.x) {
    float dot = 0.0f;

    const bf16* kn = k_nope + (j * heads + head) * nope_dim;
    for (int64_t d = 0; d < nope_dim; ++d) dot += ToF32(qn[d]) * ToF32(kn[d]);

    // No head index on the RoPE key: one per token, shared by every head.
    const bf16* kr = k_rope + j * rope_dim;
    for (int64_t d = 0; d < rope_dim; ++d) dot += ToF32(qr[d]) * ToF32(kr[d]);

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

  // Pass 2: the softmax denominator and the weighted V, accumulated together.
  // Every thread keeps a partial V vector in registers only for the columns it
  // owns, so the accumulation is over `visible` rather than over `v_dim`.
  float local_sum = 0.0f;

  extern __shared__ float acc[];  // v_dim floats, zeroed below
  for (int64_t d = threadIdx.x; d < v_dim; d += blockDim.x) acc[d] = 0.0f;
  __syncthreads();

  for (int64_t j = 0; j < visible; ++j) {
    // The score, recomputed rather than stored: keeping `visible` scores would
    // need a scratch allocation sized by context length, and this kernel is
    // the correctness reference rather than the fast path.
    float dot = 0.0f;

    const bf16* kn = k_nope + (j * heads + head) * nope_dim;
    for (int64_t d = threadIdx.x; d < nope_dim; d += blockDim.x) {
      dot += ToF32(qn[d]) * ToF32(kn[d]);
    }

    const bf16* kr = k_rope + j * rope_dim;
    for (int64_t d = threadIdx.x; d < rope_dim; d += blockDim.x) {
      dot += ToF32(qr[d]) * ToF32(kr[d]);
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

    const bf16* vj = v + (j * heads + head) * v_dim;
    for (int64_t d = threadIdx.x; d < v_dim; d += blockDim.x) {
      acc[d] += weight * ToF32(vj[d]);
    }
    __syncthreads();
  }

  __shared__ float denom;
  if (threadIdx.x == 0) denom = local_sum;
  __syncthreads();

  const float inv = 1.0f / denom;
  bf16* dst = out + (query * heads + head) * v_dim;
  for (int64_t d = threadIdx.x; d < v_dim; d += blockDim.x) {
    dst[d] = ToBf16(acc[d] * inv);
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

Status MlaRopeInPlace(const TensorView& x, int64_t rope_dim,
                      const TensorView& positions, float theta,
                      cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(x, DataType::kBFloat16, 3, "x"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(positions, DataType::kInt32, 1, "positions"));

  const int64_t tokens = x.Dim(0);
  const int64_t heads = x.Dim(1);
  const int64_t head_dim = x.Dim(2);

  if (rope_dim <= 0 || rope_dim > head_dim || rope_dim % 2 != 0) {
    return InvalidArgumentError("rope_dim must be even and within head_dim (",
                                head_dim, "), got ", rope_dim);
  }

  if (positions.Dim(0) != tokens) {
    return InvalidArgumentError("positions has ", positions.Dim(0),
                                " entries but x has ", tokens, " tokens");
  }

  if (tokens == 0) return OkStatus();

  const dim3 grid(static_cast<unsigned>(tokens), static_cast<unsigned>(heads));

  RopeInPlaceKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<bf16*>(x.Data()),
      static_cast<const int32_t*>(positions.Data()), heads, head_dim, rope_dim,
      theta);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status SplitTrailing(const TensorView& src, const TensorView& head,
                     const TensorView& tail, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(src, DataType::kBFloat16, 3, "src"));
  INFERX_RETURN_IF_ERROR(CheckTensor(head, DataType::kBFloat16, 3, "head"));
  INFERX_RETURN_IF_ERROR(CheckTensor(tail, DataType::kBFloat16, 3, "tail"));

  const int64_t rows = src.Dim(0);
  const int64_t heads = src.Dim(1);
  const int64_t head_width = head.Dim(2);
  const int64_t tail_width = tail.Dim(2);

  if (head.Dim(0) != rows || tail.Dim(0) != rows || head.Dim(1) != heads ||
      tail.Dim(1) != heads) {
    return InvalidArgumentError("SplitTrailing: src ",
                                src.GetShape().ToString(), " does not match "
                                "head ", head.GetShape().ToString(), " and "
                                "tail ", tail.GetShape().ToString());
  }

  if (head_width + tail_width != src.Dim(2)) {
    return InvalidArgumentError("SplitTrailing: ", head_width, " + ",
                                tail_width, " does not cover ", src.Dim(2));
  }

  if (rows == 0) return OkStatus();

  const dim3 grid(static_cast<unsigned>(rows), static_cast<unsigned>(heads));

  SplitTrailingKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const bf16*>(src.Data()), static_cast<bf16*>(head.Data()),
      static_cast<bf16*>(tail.Data()), heads, head_width, tail_width);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status MlaAppendLatent(const TensorView& latent, const TensorView& rope_key,
                       const TensorView& cache, const TensorView& slot_mapping,
                       cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(latent, DataType::kBFloat16, 2, "latent"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(rope_key, DataType::kBFloat16, 2, "rope_key"));
  INFERX_RETURN_IF_ERROR(CheckTensor(cache, DataType::kBFloat16, 4, "cache"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(slot_mapping, DataType::kInt32, 1, "slot_mapping"));

  const int64_t tokens = latent.Dim(0);
  const int64_t latent_dim = latent.Dim(1);
  const int64_t rope_dim = rope_key.Dim(1);

  if (rope_key.Dim(0) != tokens || slot_mapping.Dim(0) != tokens) {
    return InvalidArgumentError("latent has ", tokens, " tokens but rope_key "
                                "has ", rope_key.Dim(0), " and slot_mapping "
                                "has ", slot_mapping.Dim(0));
  }

  // The pool hands MLA a [blocks, block_size, 1, width] view: one entry per
  // token, one "head", and the latent plus its RoPE key as that head's width.
  if (cache.Dim(2) != 1 || cache.Dim(3) != latent_dim + rope_dim) {
    return InvalidArgumentError(
        "cache is ", cache.GetShape().ToString(), " but an MLA latent needs [",
        cache.Dim(0), ", ", cache.Dim(1), ", 1, ", latent_dim + rope_dim, "]");
  }

  if (tokens == 0) return OkStatus();

  AppendLatentKernel<<<static_cast<int>(tokens), kBlock, 0, stream>>>(
      static_cast<const bf16*>(latent.Data()),
      static_cast<const bf16*>(rope_key.Data()),
      static_cast<bf16*>(cache.Data()),
      static_cast<const int32_t*>(slot_mapping.Data()), latent_dim, rope_dim);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status MlaGatherLatents(const TensorView& cache, const TensorView& block_table,
                        int64_t context_len, const TensorView& out,
                        cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(cache, DataType::kBFloat16, 4, "cache"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(block_table, DataType::kInt32, 1, "block_table"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));

  const int64_t block_size = cache.Dim(1);
  const int64_t width = cache.Dim(3);

  if (out.Dim(0) < context_len || out.Dim(1) != width) {
    return InvalidArgumentError("out is ", out.GetShape().ToString(),
                                " but ", context_len, " x ", width,
                                " was asked for");
  }

  const int64_t blocks_needed = (context_len + block_size - 1) / block_size;
  if (block_table.Dim(0) < blocks_needed) {
    return InvalidArgumentError("block_table holds ", block_table.Dim(0),
                                " blocks but ", context_len,
                                " tokens need ", blocks_needed);
  }

  if (context_len == 0) return OkStatus();

  GatherLatentsKernel<<<static_cast<int>(context_len), kBlock, 0, stream>>>(
      static_cast<const bf16*>(cache.Data()),
      static_cast<const int32_t*>(block_table.Data()),
      static_cast<bf16*>(out.Data()), width, block_size);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status MlaAttention(const TensorView& q_nope, const TensorView& q_rope,
                    const TensorView& k_nope, const TensorView& k_rope,
                    const TensorView& v, const TensorView& out,
                    int64_t query_base, float scale, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(q_nope, DataType::kBFloat16, 3, "q_nope"));
  INFERX_RETURN_IF_ERROR(CheckTensor(q_rope, DataType::kBFloat16, 3, "q_rope"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k_nope, DataType::kBFloat16, 3, "k_nope"));
  INFERX_RETURN_IF_ERROR(CheckTensor(k_rope, DataType::kBFloat16, 2, "k_rope"));
  INFERX_RETURN_IF_ERROR(CheckTensor(v, DataType::kBFloat16, 3, "v"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 3, "out"));

  const int64_t q_tokens = q_nope.Dim(0);
  const int64_t heads = q_nope.Dim(1);
  const int64_t nope_dim = q_nope.Dim(2);
  const int64_t rope_dim = q_rope.Dim(2);
  const int64_t context = k_nope.Dim(0);
  const int64_t v_dim = v.Dim(2);

  if (q_rope.Dim(0) != q_tokens || q_rope.Dim(1) != heads) {
    return InvalidArgumentError("q_rope is ", q_rope.GetShape().ToString(),
                                " but q_nope is ",
                                q_nope.GetShape().ToString());
  }

  if (k_nope.Dim(1) != heads || k_nope.Dim(2) != nope_dim) {
    return InvalidArgumentError("k_nope is ", k_nope.GetShape().ToString(),
                                " but q_nope is ",
                                q_nope.GetShape().ToString());
  }

  if (k_rope.Dim(0) != context || k_rope.Dim(1) != rope_dim) {
    return InvalidArgumentError("k_rope must be [", context, ", ", rope_dim,
                                "], got ", k_rope.GetShape().ToString());
  }

  if (v.Dim(0) != context || v.Dim(1) != heads) {
    return InvalidArgumentError("v is ", v.GetShape().ToString(),
                                " but the context is ", context,
                                " over ", heads, " heads");
  }

  if (out.Dim(0) != q_tokens || out.Dim(1) != heads || out.Dim(2) != v_dim) {
    return InvalidArgumentError("out must be [", q_tokens, ", ", heads, ", ",
                                v_dim, "], got ", out.GetShape().ToString());
  }

  // Causality has to be satisfiable: the last query must see at least itself.
  if (query_base < 0 || query_base + q_tokens > context) {
    return InvalidArgumentError("query_base ", query_base, " with ", q_tokens,
                                " queries runs past a context of ", context);
  }

  if (q_tokens == 0) return OkStatus();

  const dim3 grid(static_cast<unsigned>(q_tokens),
                  static_cast<unsigned>(heads));
  const size_t shared = static_cast<size_t>(v_dim) * sizeof(float);

  MlaAttentionKernel<<<grid, kBlock, shared, stream>>>(
      static_cast<const bf16*>(q_nope.Data()),
      static_cast<const bf16*>(q_rope.Data()),
      static_cast<const bf16*>(k_nope.Data()),
      static_cast<const bf16*>(k_rope.Data()),
      static_cast<const bf16*>(v.Data()), static_cast<bf16*>(out.Data()),
      heads, nope_dim, rope_dim, v_dim, context, query_base, scale);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace inferx::kernels
