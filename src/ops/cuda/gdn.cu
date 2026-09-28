#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/ops/cuda/gdn.h"

namespace inferx::ops::cuda {
namespace {

constexpr int kThreads = 256;
constexpr int kDvTile = 32;  // Value columns one recurrent block owns.

__device__ __forceinline__ float Silu(float x) { return x / (1.0f + __expf(-x)); }

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

/// Source rows per key-head group: [q | k | v | z] with widths
/// kd, kd, nvg * vd, nvg * vd. Destinations are block-contiguous.
__global__ void SplitGdnKernel(const __nv_bfloat16* __restrict__ packed,
                               __nv_bfloat16* __restrict__ conv_in,
                               __nv_bfloat16* __restrict__ z, int64_t rows, int groups,
                               int kd, int nvg, int vd) {
  const int64_t group_width = 2 * kd + 2 * nvg * vd;
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * groups * group_width) return;
  const int64_t row = i / (groups * group_width);
  const int64_t rest = i % (groups * group_width);
  const int64_t group = rest / group_width;
  const int64_t col = rest % group_width;
  const __nv_bfloat16 value = packed[i];
  const int64_t q_total = int64_t(groups) * kd;
  const int64_t v_total = int64_t(groups) * nvg * vd;
  if (col < 2 * kd) {
    // Q or K stay in order inside [q | k].
    const int64_t dest = row * (q_total * 2 + v_total) + col < q_total
                             ? row * (2 * q_total + v_total) + col
                             : row * (2 * q_total + v_total) + q_total + (col - kd);
    conv_in[dest] = value;
  } else if (col < 2 * kd + nvg * vd) {
    const int64_t dest = row * (2 * q_total + v_total) + 2 * q_total +
                         group * nvg * vd + (col - 2 * kd);
    conv_in[dest] = value;
  } else {
    const int64_t dest =
        row * v_total + group * nvg * vd + (col - 2 * kd - nvg * vd);
    z[dest] = value;
  }
}

/// One thread owns one channel of one sequence and walks its tokens in
/// order, keeping the kernel-1 most recent inputs in the state ring.
__global__ void ConvKernel(__nv_bfloat16* __restrict__ x,
                           const __nv_bfloat16* __restrict__ weight,
                           float* __restrict__ state, const int* __restrict__ qo, int channels,
                           int kernel, int num_seqs) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) return;
  const __nv_bfloat16* w =
      reinterpret_cast<const __nv_bfloat16*>(weight) + int64_t(channel) * kernel;
  for (int seq = 0; seq < num_seqs; ++seq) {
    float* st = state + (int64_t(seq) * channels + channel) * (kernel - 1);
    const int begin = qo[seq], end = qo[seq + 1];
    for (int t = begin; t < end; ++t) {
      __nv_bfloat16* xt = x + int64_t(t) * channels + channel;
      const float current = __bfloat162float(*xt);
      float sum = __bfloat162float(w[kernel - 1]) * current;
      for (int j = 0; j < kernel - 1; ++j) sum += __bfloat162float(w[j]) * st[j];
      for (int j = 0; j + 1 < kernel - 1; ++j) st[j] = st[j + 1];
      if (kernel > 1) st[kernel - 2] = current;
      *xt = __float2bfloat16(Silu(sum));
    }
  }
}

__global__ void GatesKernel(const __nv_bfloat16* __restrict__ ba,
                            const float* __restrict__ a_log,
                            const float* __restrict__ dt_bias, float* __restrict__ beta,
                            float* __restrict__ g, int64_t rows, int value_heads, int nvg) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t per_group = 2 * nvg;
  if (i >= rows * int64_t(value_heads)) return;
  const int64_t row = i / value_heads;
  const int head = i % value_heads;
  // Per-group [b (nvg) | a (nvg)] layout.
  const int64_t base = row * int64_t(value_heads) * 2 + (head / nvg) * per_group;
  const int lane = head % nvg;
  const float b = __bfloat162float(ba[base + lane]);
  const float a = __bfloat162float(ba[base + nvg + lane]);
  beta[i] = 1.0f / (1.0f + __expf(-b));
  const float sp = a + dt_bias[head] > 20.0f
                       ? a + dt_bias[head]
                       : logf(1.0f + __expf(a + dt_bias[head]));
  g[i] = -__expf(a_log[head]) * sp;
}

/// One block per (sequence, value head, dv tile); warps within the block
/// split the tile's columns, and each warp reduces over dk with shuffles.
/// S[:, tile] lives in shared memory; the delta correction touches only
/// this tile's columns, so tiles never exchange.
__global__ void RecurrentKernel(const __nv_bfloat16* __restrict__ conv,
                                const float* __restrict__ beta, const float* __restrict__ g,
                                float* __restrict__ state, const int* __restrict__ slots,
                                const int* __restrict__ qo, __nv_bfloat16* __restrict__ y,
                                int conv_width, int heads_ratio, int dk, int dv,
                                int value_heads) {
  const int seq = blockIdx.x;
  const int head = blockIdx.y;
  const int tile = blockIdx.z;
  const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32, warps = blockDim.x / 32;
  const int col0 = tile * kDvTile;
  const int cols = min(kDvTile, dv - col0);
  extern __shared__ float smem[];
  float* s = smem;                    // [dk * kDvTile] state tile.
  float* kvec = smem + dk * kDvTile;  // [dk]
  float* qvec = kvec + dk;            // [dk]

  float* st = state + ((int64_t(slots[seq]) * value_heads + head) * dk) * dv;
  for (int i = tid; i < dk * kDvTile; i += blockDim.x) {
    // Columns past dv are padding for this tile; leave shared memory zero
    // so the update and output loops see real zeros there.
    const int col = col0 + (i % kDvTile);
    s[i] = col < dv ? st[(i / kDvTile) * dv + col] : 0.0f;
  }
  __syncthreads();

  const int begin = qo[seq], end = qo[seq + 1];
  const float qscale = rsqrtf(float(dk));
  const int khead = head / heads_ratio;
  const int64_t qw = (int64_t(conv_width) - int64_t(value_heads) * dv) / 2;
  for (int t = begin; t < end; ++t) {
    // L2-normalize this token's k and q heads (eps 1e-6), scaling q to
    // 1/sqrt(dk); fp32 throughout, matching the reference recurrence.
    {
      const __nv_bfloat16* krow = conv + int64_t(t) * conv_width + qw + int64_t(khead) * dk;
      const __nv_bfloat16* qrow_t = conv + int64_t(t) * conv_width + int64_t(khead) * dk;
      float kp = 0, qp = 0;
      for (int d = tid; d < dk; d += blockDim.x) {
        kp += __bfloat162float(krow[d]) * __bfloat162float(krow[d]);
        qp += __bfloat162float(qrow_t[d]) * __bfloat162float(qrow_t[d]);
      }
      __shared__ float totals[2 * (kThreads / 32)];
      for (int off = 16; off > 0; off /= 2) {
        kp += __shfl_down_sync(0xffffffffu, kp, off);
        qp += __shfl_down_sync(0xffffffffu, qp, off);
      }
      if (lane == 0) {
        totals[warp] = kp;
        totals[kThreads / 32 + warp] = qp;
      }
      __syncthreads();
      if (tid < 32) {
        float ks = 0, qs = 0;
        for (int w = 0; w < warps; ++w) {
          ks += totals[w];
          qs += totals[warps + w];
        }
        if (lane == 0) {
          totals[0] = rsqrtf(ks + 1e-6f);
          totals[1] = rsqrtf(qs + 1e-6f);
        }
      }
      __syncthreads();
      const float kn = totals[0], qn = totals[1] * qscale;
      for (int d = tid; d < dk; d += blockDim.x) {
        kvec[d] = __bfloat162float(krow[d]) * kn;
        qvec[d] = __bfloat162float(qrow_t[d]) * qn;
      }
      __syncthreads();
    }

    const float decay = __expf(g[int64_t(t) * value_heads + head]);
    const float b = beta[int64_t(t) * value_heads + head];
    for (int i = tid; i < dk * kDvTile; i += blockDim.x) s[i] *= decay;

    // Each warp owns strided columns of the tile; lanes reduce over dk.
    for (int c = warp; c < cols; c += warps) {
      float kv_mem = 0, y_sum = 0;
      for (int d = lane; d < dk; d += 32) kv_mem += s[d * kDvTile + c] * kvec[d];
      for (int off = 16; off > 0; off /= 2) kv_mem += __shfl_down_sync(~0u, kv_mem, off);
      kv_mem = __shfl_sync(~0u, kv_mem, 0);
      const float value = __bfloat162float(
          conv[int64_t(t) * conv_width + 2 * qw + int64_t(head) * dv + col0 + c]);
      const float delta = b * (value - kv_mem);
      for (int d = lane; d < dk; d += 32) {
        s[d * kDvTile + c] += kvec[d] * delta;
      }
      __syncwarp();
      for (int d = lane; d < dk; d += 32) y_sum += s[d * kDvTile + c] * qvec[d];
      for (int off = 16; off > 0; off /= 2) y_sum += __shfl_down_sync(~0u, y_sum, off);
      if (lane == 0) {
        y[int64_t(t) * value_heads * dv + int64_t(head) * dv + col0 + c] =
            __float2bfloat16(y_sum);
      }
    }
    __syncthreads();
  }
  __syncthreads();
  for (int i = tid; i < dk * kDvTile; i += blockDim.x) {
    const int col = col0 + (i % kDvTile);
    if (col < dv) st[(i / kDvTile) * dv + col] = s[i];
  }
}

__global__ void NormGatedKernel(const __nv_bfloat16* __restrict__ y,
                                const __nv_bfloat16* __restrict__ z,
                                const __nv_bfloat16* __restrict__ weight, float eps,
                                __nv_bfloat16* __restrict__ out, int64_t rows, int heads,
                                int dim) {
  const int64_t head_index = (int64_t(blockIdx.x) * blockDim.x + threadIdx.x);
  if (head_index >= rows * heads) return;
  const int64_t base = head_index * dim;
  float sum = 0;
  for (int d = 0; d < dim; ++d) {
    const float value = __bfloat162float(y[base + d]);
    sum += value * value;
  }
  const float inv = rsqrtf(sum / dim + eps);
  for (int d = 0; d < dim; ++d) {
    const float normed = __bfloat162float(y[base + d]) * inv *
                         __bfloat162float(weight[d]);
    const float gate = Silu(__bfloat162float(z[base + d]));
    out[base + d] = __float2bfloat16(normed * gate);
  }
}

}  // namespace

Status SplitGdnProjection(ExecutionContext& ctx, const Tensor& packed, Tensor& conv_in,
                          Tensor& z, int64_t key_heads, int64_t key_dim, int64_t value_heads,
                          int64_t value_dim) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int64_t nvg = value_heads / key_heads;
  const int64_t total = packed.Numel();
  SplitGdnKernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                   static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const __nv_bfloat16*>(packed.Data()),
      static_cast<__nv_bfloat16*>(conv_in.Data()), static_cast<__nv_bfloat16*>(z.Data()),
      packed.Dim(0), static_cast<int>(key_heads), static_cast<int>(key_dim),
      static_cast<int>(nvg), static_cast<int>(value_dim));
  return CudaError(cudaGetLastError(), "split gdn projection");
}

Status GdnCausalConv(ExecutionContext& ctx, Tensor& x, const Tensor& weight,
                     const Tensor& state, const Tensor& batch_indices,
                     const Tensor& qo_indptr, int64_t kernel) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int channels = static_cast<int>(x.Dim(1));
  const int num_seqs = static_cast<int>(qo_indptr.Numel() - 1);
  ConvKernel<<<(channels + kThreads - 1) / kThreads, kThreads, 0,
               static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<__nv_bfloat16*>(x.Data()),
      static_cast<const __nv_bfloat16*>(weight.Data()),
      static_cast<float*>(state.Data()),
      static_cast<const int*>(qo_indptr.Data()), channels, static_cast<int>(kernel),
      num_seqs);
  return CudaError(cudaGetLastError(), "gdn causal conv");
}

Status GdnGates(ExecutionContext& ctx, const Tensor& ba, const Tensor& a_log,
                const Tensor& dt_bias, Tensor& beta, Tensor& g) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int64_t rows = ba.Dim(0);
  const int64_t heads = beta.Numel() / rows;
  const int64_t nvg = ba.Dim(1) / 2 / heads;  // b/a pairs per head, grouped.
  GatesKernel<<<static_cast<uint32_t>((rows * heads + kThreads - 1) / kThreads), kThreads, 0,
                static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const __nv_bfloat16*>(ba.Data()), static_cast<const float*>(a_log.Data()),
      static_cast<const float*>(dt_bias.Data()), static_cast<float*>(beta.Data()),
      static_cast<float*>(g.Data()), rows, static_cast<int>(heads), static_cast<int>(nvg));
  return CudaError(cudaGetLastError(), "gdn gates");
}

Status GdnRecurrent(ExecutionContext& ctx, const Tensor& conv, int64_t query_width,
                    const Tensor& beta, const Tensor& g, Tensor& state,
                    const Tensor& slot_indices, const Tensor& qo_indptr,
                    const Tensor& batch_indices, Tensor& y) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int num_seqs = static_cast<int>(qo_indptr.Numel() - 1);
  const int dv = static_cast<int>(state.Dim(3));
  const int dk = static_cast<int>(state.Dim(2));
  const int value_heads = static_cast<int>(state.Dim(1));
  const int key_heads = static_cast<int>(query_width / dk);
  const int tiles = (dv + kDvTile - 1) / kDvTile;
  dim3 grid(num_seqs, value_heads, tiles);
  const int smem = (dk * kDvTile + 2 * dk) * sizeof(float);
  if (dk > kThreads) return UnimplementedError("gdn recurrent supports key dims up to 256");
  RecurrentKernel<<<grid, kThreads, smem, static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const __nv_bfloat16*>(conv.Data()),
      static_cast<const float*>(beta.Data()), static_cast<const float*>(g.Data()),
      static_cast<float*>(state.Data()), static_cast<const int*>(slot_indices.Data()),
      static_cast<const int*>(qo_indptr.Data()), static_cast<__nv_bfloat16*>(y.Data()),
      static_cast<int>(conv.Dim(1)), value_heads / key_heads, dk, dv, value_heads);
  return CudaError(cudaGetLastError(), "gdn recurrent");
}

Status RmsNormGated(ExecutionContext& ctx, const Tensor& y, const Tensor& z,
                    const Tensor& weight, float eps, Tensor& out) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int64_t heads_total = y.Dim(0) * (y.Dim(1) / weight.Dim(0));
  NormGatedKernel<<<static_cast<uint32_t>((heads_total + kThreads - 1) / kThreads), kThreads,
                    0, static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const __nv_bfloat16*>(y.Data()),
      static_cast<const __nv_bfloat16*>(z.Data()),
      static_cast<const __nv_bfloat16*>(weight.Data()), eps,
      static_cast<__nv_bfloat16*>(out.Data()), y.Dim(0),
      static_cast<int>(y.Dim(1) / weight.Dim(0)), static_cast<int>(weight.Dim(0)));
  return CudaError(cudaGetLastError(), "rms norm gated");
}

}  // namespace inferx::ops::cuda
