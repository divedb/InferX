#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/ops/cuda/moe.h"

namespace inferx::ops::cuda {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float LogitDot(const __nv_bfloat16* x, const __nv_bfloat16* w,
                                          int hidden, int lane) {
  // One warp owns one expert: lanes stride the hidden dimension and the
  // warp shuffles reduce, so experts never share a reduction.
  float partial = 0.0f;
  for (int d = lane; d < hidden; d += 32) {
    partial += __bfloat162float(x[d]) * __bfloat162float(w[d]);
  }
  for (int offset = 16; offset > 0; offset /= 2) {
    partial += __shfl_down_sync(0xffffffffu, partial, offset);
  }
  return partial;
}

/// \brief One block per token row.
///
/// Thread e < num_experts computes that expert's logit (plus optional bias),
/// then the block runs the scoring policy: top-k over the (possibly biased
/// or sigmoid-transformed) scores, weights from the chosen variant, and the
/// optional renormalization and routed scale. Deterministic: selection is a
/// scan with a fixed tie-break toward the lower index.
__global__ void RouteTokensKernel(const __nv_bfloat16* __restrict__ hidden,
                                  const __nv_bfloat16* __restrict__ weight,
                                  const __nv_bfloat16* __restrict__ bias,
                                  const float* __restrict__ correction, int rows, int hidden_dim,
                                  int experts, RoutingConfig config, int* __restrict__ indices,
                                  float* __restrict__ weights) {
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  __shared__ float logits[1024];
  const int warp = tid / 32, lane = tid % 32, warps = blockDim.x / 32;
  for (int e = warp; e < experts; e += warps) {
    float logit = LogitDot(hidden + int64_t(row) * hidden_dim,
                           weight + int64_t(e) * hidden_dim, hidden_dim, lane);
    if (bias != nullptr) logit += __bfloat162float(bias[e]);
    if (lane == 0) logits[e] = logit;
  }
  __syncthreads();
  if (tid != 0) return;

  float scores[512];
  bool selectable[512];
  for (int e = 0; e < experts; ++e) {
    scores[e] = config.scoring == RouterScoring::kSigmoidGroupTopk
                    ? 1.0f / (1.0f + expf(-logits[e]))
                    : logits[e];
    selectable[e] = true;
  }
  if (config.scoring == RouterScoring::kSigmoidGroupTopk && config.group_count > 1) {
    // Grouped selection: rank groups by the sum of their top-2 biased
    // scores, keep the top group_topk groups, and mask everyone else out
    // of the final top-k. Group scores live past the expert logits.
    const int per_group = experts / config.group_count;
    float* group_scores = logits + experts;
    for (int g = 0; g < config.group_count; ++g) {
      float best = -INFINITY, second = -INFINITY;
      for (int i = 0; i < per_group; ++i) {
        float s = scores[g * per_group + i];
        if (correction != nullptr) s += correction[g * per_group + i];
        if (s > best) {
          second = best;
          best = s;
        } else if (s > second) {
          second = s;
        }
      }
      group_scores[g] = best + second;
    }
    for (int kept = 0; kept < config.group_topk; ++kept) {
      int best_g = -1;
      float best = -INFINITY;
      for (int g = 0; g < config.group_count; ++g) {
        if (group_scores[g] > best) {
          best = group_scores[g];
          best_g = g;
        }
      }
      if (best_g < 0) break;
      group_scores[best_g] = -INFINITY;
      for (int i = 0; i < per_group; ++i) selectable[best_g * per_group + i] = true;
    }
  }

  float selected[64];
  int chosen[64];
  for (int k = 0; k < config.topk; ++k) {
    int best_e = -1;
    float best = -INFINITY;
    for (int e = 0; e < experts; ++e) {
      float s = scores[e];
      if (correction != nullptr && config.scoring == RouterScoring::kSigmoidGroupTopk) {
        s += correction[e];
      }
      if (selectable[e] && s > best) {
        best = s;
        best_e = e;
      }
    }
    if (best_e < 0) break;
    selectable[best_e] = false;
    chosen[k] = best_e;
    selected[k] =
        config.scoring == RouterScoring::kSigmoidGroupTopk ? scores[best_e] : logits[best_e];
  }

  if (config.scoring == RouterScoring::kSoftmaxTopkRenorm) {
    // Softmax over the selected logits == renormalized full softmax.
    float max_logit = -INFINITY;
    for (int k = 0; k < config.topk; ++k) max_logit = fmaxf(max_logit, selected[k]);
    float sum = 0.0f;
    for (int k = 0; k < config.topk; ++k) {
      selected[k] = expf(selected[k] - max_logit);
      sum += selected[k];
    }
    for (int k = 0; k < config.topk; ++k) selected[k] /= sum;
  } else if (config.normalize) {
    float sum = 0.0f;
    for (int k = 0; k < config.topk; ++k) sum += selected[k];
    for (int k = 0; k < config.topk; ++k) selected[k] /= (sum + 1e-20f);
  }
  for (int k = 0; k < config.topk; ++k) {
    selected[k] *= config.routing_scale;
    indices[int64_t(row) * config.topk + k] = chosen[k];
    weights[int64_t(row) * config.topk + k] = selected[k];
  }
}

__global__ void HistogramKernel(const int* __restrict__ indices, int64_t total,
                                int* __restrict__ counts) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  atomicAdd(&counts[indices[i]], 1);
}

__global__ void ScatterSlotsKernel(const int* __restrict__ indices,
                                   const float* __restrict__ weights, const int* offsets,
                                   int* cursor, int topk, int64_t total, int* token_rows,
                                   float* weights_by_slot) {
  const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= total) return;
  const int expert = indices[i];
  const int64_t slot = offsets[expert] + atomicAdd(&cursor[expert], 1);
  token_rows[slot] = static_cast<int>(i / topk);
  weights_by_slot[slot] = weights[i];
}

__global__ void GatherKernel(const __nv_bfloat16* __restrict__ hidden,
                             const int* __restrict__ token_rows, int hidden_dim,
                             int64_t slots, __nv_bfloat16* __restrict__ out) {
  const int64_t slot = blockIdx.x;
  if (slot >= slots) return;
  const __nv_bfloat16* row = hidden + int64_t(token_rows[slot]) * hidden_dim;
  __nv_bfloat16* dst = out + slot * hidden_dim;
  for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) dst[d] = row[d];
}

__global__ void ScatterAddKernel(const __nv_bfloat16* __restrict__ expert_rows,
                                 const int* __restrict__ token_rows,
                                 const float* __restrict__ weights_by_slot, int hidden_dim,
                                 int64_t count, __nv_bfloat16* __restrict__ out) {
  const int64_t slot = blockIdx.x;
  if (slot >= count) return;
  const float w = weights_by_slot[slot];
  const __nv_bfloat16* src = expert_rows + slot * hidden_dim;
  __nv_bfloat16* dst = out + int64_t(token_rows[slot]) * hidden_dim;
  for (int d = threadIdx.x; d < hidden_dim; d += blockDim.x) {
    dst[d] = __float2bfloat16(__bfloat162float(dst[d]) + w * __bfloat162float(src[d]));
  }
}

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

}  // namespace

Status RouteTokens(ExecutionContext& ctx, const Tensor& hidden, const Tensor& router_weight,
                   const Tensor* router_bias, const Tensor* correction_bias,
                   const RoutingConfig& config, Tensor& topk_indices, Tensor& topk_weights) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  const int experts = router_weight.Dim(0);
  if (experts > 512) return UnimplementedError("router kernel supports up to 512 experts");
  if (config.topk > 64) return UnimplementedError("router kernel supports up to 64 selected");
  RouteTokensKernel<<<static_cast<uint32_t>(hidden.Dim(0)), kThreads, 0,
                      static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const __nv_bfloat16*>(hidden.Data()),
      static_cast<const __nv_bfloat16*>(router_weight.Data()),
      router_bias == nullptr ? nullptr
                             : static_cast<const __nv_bfloat16*>(router_bias->Data()),
      correction_bias == nullptr ? nullptr : static_cast<const float*>(correction_bias->Data()),
      static_cast<int>(hidden.Dim(0)), static_cast<int>(hidden.Dim(1)), experts, config,
      static_cast<int*>(topk_indices.Data()), static_cast<float*>(topk_weights.Data()));
  return CudaError(cudaGetLastError(), "route tokens");
}

Status ExpertHistogram(ExecutionContext& ctx, const Tensor& topk_indices, int64_t num_experts,
                       Tensor& counts) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream());
  INFERX_RETURN_IF_ERROR(CudaError(
      cudaMemsetAsync(counts.Data(), 0, num_experts * sizeof(int), stream), "zero counts"));
  const int64_t total = topk_indices.Numel();
  HistogramKernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                    stream>>>(static_cast<const int*>(topk_indices.Data()), total,
                              static_cast<int*>(counts.Data()));
  return CudaError(cudaGetLastError(), "expert histogram");
}

Status ScatterSlots(ExecutionContext& ctx, const Tensor& topk_indices, const Tensor& topk_weights,
                    const Tensor& offsets, Tensor& cursor, Tensor& token_rows,
                    Tensor& weights_by_slot) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  cudaStream_t stream = static_cast<cudaStream_t>(ctx.stream());
  const int64_t experts = offsets.Numel() - 1;
  INFERX_RETURN_IF_ERROR(CudaError(
      cudaMemsetAsync(cursor.Data(), 0, experts * sizeof(int), stream), "zero cursor"));
  const int64_t total = topk_indices.Numel();
  ScatterSlotsKernel<<<static_cast<uint32_t>((total + kThreads - 1) / kThreads), kThreads, 0,
                       stream>>>(
      static_cast<const int*>(topk_indices.Data()),
      static_cast<const float*>(topk_weights.Data()), static_cast<const int*>(offsets.Data()),
      static_cast<int*>(cursor.Data()), static_cast<int>(topk_indices.Dim(1)), total,
      static_cast<int*>(token_rows.Data()), static_cast<float*>(weights_by_slot.Data()));
  return CudaError(cudaGetLastError(), "scatter slots");
}

Status GatherRoutedTokens(ExecutionContext& ctx, const Tensor& hidden, const Tensor& token_rows,
                          Tensor& out) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  GatherKernel<<<static_cast<uint32_t>(token_rows.Numel()), kThreads, 0,
                 static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const __nv_bfloat16*>(hidden.Data()),
      static_cast<const int*>(token_rows.Data()), static_cast<int>(hidden.Dim(1)),
      token_rows.Numel(), static_cast<__nv_bfloat16*>(out.Data()));
  return CudaError(cudaGetLastError(), "gather routed tokens");
}

Status ScatterRoutedOutputs(ExecutionContext& ctx, const Tensor& expert_rows,
                            const Tensor& token_rows, const Tensor& weights_by_slot,
                            int64_t begin, int64_t count, Tensor& out) {
  INFERX_RETURN_IF_ERROR(ctx.runtime().Activate());
  ScatterAddKernel<<<static_cast<uint32_t>(count), kThreads, 0,
                     static_cast<cudaStream_t>(ctx.stream())>>>(
      static_cast<const __nv_bfloat16*>(expert_rows.Data()) + begin * expert_rows.Dim(1),
      static_cast<const int*>(token_rows.Data()) + begin,
      static_cast<const float*>(weights_by_slot.Data()) + begin,
      static_cast<int>(expert_rows.Dim(1)), count, static_cast<__nv_bfloat16*>(out.Data()));
  return CudaError(cudaGetLastError(), "scatter routed outputs");
}

}  // namespace inferx::ops::cuda
