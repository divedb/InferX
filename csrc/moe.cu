#include "inferx/kernels/moe.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"

namespace inferx::kernels {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kBlock = 256;

// The largest k a token can be routed to. Real architectures use 2 (Mixtral),
// 4 (Qwen2-MoE) or 8 (DeepSeek-V3); the cap exists so the router can hold a
// token's chosen experts in registers rather than in a scratch allocation, and
// it is checked on the host so exceeding it is an error rather than a silently
// truncated mixture.
constexpr int kMaxTopK = 16;

// Experts one router block can score. E is a few dozen to a few hundred in
// every architecture we know of, and holding the row in shared memory is what
// makes the softmax and the k selection passes over it free.
constexpr int kMaxExperts = 1024;

// One block per token. Softmax across experts in fp32, then k rounds of
// masked argmax.
//
// A full sort would also work and is what a large k would want, but k is at
// most a handful and k passes over a row already in shared memory is cheaper
// than sorting E of them. The masked-argmax form is also what makes the
// tie-break rule expressible: strictly-greater comparison keeps the lowest
// index among equals, so equal logits route the same way on every run.
__global__ void RouteTopKKernel(const bf16* __restrict__ logits,
                                float* __restrict__ out_weights,
                                int32_t* __restrict__ out_experts,
                                int64_t num_experts, int k, bool renormalize,
                                float gate_scale) {
  extern __shared__ float row[];

  const int64_t token = blockIdx.x;
  const bf16* src = logits + token * num_experts;

  // Pass 1: the row's maximum, for a softmax that cannot overflow.
  float local_max = -INFINITY;
  for (int64_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
    const float v = __bfloat162float(src[e]);
    row[e] = v;
    local_max = fmaxf(local_max, v);
  }

  __shared__ float reduce[kBlock];
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

  // Pass 2: exponentiate in place and sum.
  float local_sum = 0.0f;
  for (int64_t e = threadIdx.x; e < num_experts; e += blockDim.x) {
    const float v = __expf(row[e] - row_max);
    row[e] = v;
    local_sum += v;
  }

  reduce[threadIdx.x] = local_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }

  const float inv_sum = 1.0f / reduce[0];
  __syncthreads();

  // Pass 3: k rounds of argmax, one thread doing the selection.
  //
  // Serial on purpose. A block-wide argmax per round would need a reduction
  // that carries the index alongside the value and breaks ties the same way in
  // every tree position, and for E in the hundreds one thread scanning shared
  // memory k times is a few thousand cycles against a GEMM that is about to
  // run for a hundred times longer.
  if (threadIdx.x == 0) {
    float chosen[kMaxTopK];
    float total = 0.0f;

    for (int slot = 0; slot < k; ++slot) {
      int best = -1;
      float best_value = -INFINITY;

      for (int64_t e = 0; e < num_experts; ++e) {
        // Strictly greater: among equal scores the lowest index wins, so the
        // routing is a function of the logits and not of the scan order.
        if (row[e] > best_value) {
          best_value = row[e];
          best = static_cast<int>(e);
        }
      }

      chosen[slot] = best_value * inv_sum;
      total += chosen[slot];
      out_experts[token * k + slot] = best;

      // Mask it out of the next round. -INFINITY rather than 0 because a
      // softmax weight of exactly 0 is reachable and would be picked again.
      row[best] = -INFINITY;
    }

    const float scale = (renormalize && total > 0.0f) ? 1.0f / total : 1.0f;
    for (int slot = 0; slot < k; ++slot) {
      out_weights[token * k + slot] = chosen[slot] * scale * gate_scale;
    }
  }
}

// One block per expert: count the assignments that chose it.
__global__ void CountPerExpertKernel(const int32_t* __restrict__ experts,
                                     int64_t num_assignments,
                                     int32_t* __restrict__ counts) {
  const int expert = blockIdx.x;

  int local = 0;
  for (int64_t a = threadIdx.x; a < num_assignments; a += blockDim.x) {
    if (experts[a] == expert) ++local;
  }

  __shared__ int reduce[kBlock];
  reduce[threadIdx.x] = local;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) reduce[threadIdx.x] += reduce[threadIdx.x + stride];
    __syncthreads();
  }

  if (threadIdx.x == 0) counts[expert] = reduce[0];
}

// Exclusive scan of the per-expert counts into offsets, in place, in one
// thread.
//
// In place: the counts arrive in `offsets[0..E)` and are replaced by the
// offsets as the scan passes over them. One thread reading `offsets[e]` before
// writing it is what makes that safe, and it is why this is deliberately not
// parallel — the alternative is a scratch buffer, and a per-call allocation on
// the decode path would mean either a `cudaMalloc` in the steady state (§6.1
// forbids it) or a stream synchronize (which would make the whole MoE FFN
// uncapturable as a CUDA graph). E is a few hundred at most, against a launch
// this kernel is already paying for.
__global__ void ScanOffsetsKernel(int32_t* __restrict__ offsets,
                                  int64_t num_experts) {
  if (threadIdx.x != 0) return;

  int32_t running = 0;
  for (int64_t e = 0; e < num_experts; ++e) {
    const int32_t count = offsets[e];
    offsets[e] = running;
    running += count;
  }
  offsets[num_experts] = running;
}

// One block per expert, walking the assignment array in order and appending
// matches. Stable by construction: a block processes tiles left to right and,
// within a tile, a thread's position is its rank among matching threads with a
// lower index. So expert e's rows come out in ascending assignment order on
// every run, whatever the block scheduler does.
__global__ void ScatterByExpertKernel(const int32_t* __restrict__ experts,
                                      int64_t num_assignments, int k,
                                      const int32_t* __restrict__ offsets,
                                      int32_t* __restrict__ rows,
                                      int32_t* __restrict__ dest) {
  const int expert = blockIdx.x;

  __shared__ int prefix[kBlock];
  __shared__ int base;

  if (threadIdx.x == 0) base = offsets[expert];
  __syncthreads();

  for (int64_t tile = 0; tile < num_assignments; tile += blockDim.x) {
    const int64_t a = tile + threadIdx.x;
    const bool mine = a < num_assignments && experts[a] == expert;

    prefix[threadIdx.x] = mine ? 1 : 0;
    __syncthreads();

    // Inclusive scan over the tile (Hillis-Steele). blockDim.x is a power of
    // two, which is what makes the doubling stride exact.
    for (int stride = 1; stride < blockDim.x; stride *= 2) {
      const int add = threadIdx.x >= stride ? prefix[threadIdx.x - stride] : 0;
      __syncthreads();
      prefix[threadIdx.x] += add;
      __syncthreads();
    }

    if (mine) {
      const int slot_index = base + prefix[threadIdx.x] - 1;
      rows[slot_index] = static_cast<int32_t>(a / k);
      dest[a] = slot_index;
    }

    __syncthreads();
    if (threadIdx.x == blockDim.x - 1) base += prefix[threadIdx.x];
    __syncthreads();
  }
}

__global__ void GatherRowsKernel(const bf16* __restrict__ x,
                                 const int32_t* __restrict__ rows,
                                 bf16* __restrict__ out, int64_t width) {
  const int64_t i = blockIdx.x;
  const bf16* src = x + static_cast<int64_t>(rows[i]) * width;
  bf16* dst = out + i * width;

  for (int64_t c = threadIdx.x; c < width; c += blockDim.x) dst[c] = src[c];
}

// One block per token, summing that token's k expert outputs in slot order.
//
// Slot order is descending gate weight, which is both deterministic and the
// numerically better order for a sum of terms of very different magnitude.
__global__ void CombineRowsKernel(const bf16* __restrict__ y,
                                  const int32_t* __restrict__ dest,
                                  const float* __restrict__ weights,
                                  bf16* __restrict__ out, int64_t width,
                                  int k) {
  const int64_t token = blockIdx.x;

  for (int64_t c = threadIdx.x; c < width; c += blockDim.x) {
    float acc = 0.0f;

    for (int slot = 0; slot < k; ++slot) {
      const int64_t a = token * k + slot;
      const int64_t row = dest[a];
      acc += weights[a] * __bfloat162float(y[row * width + c]);
    }

    out[token * width + c] = __float2bfloat16(acc);
  }
}

__global__ void AddSharedExpertKernel(const bf16* __restrict__ shared,
                                      const bf16* __restrict__ gate_logits,
                                      bf16* __restrict__ out, int64_t width) {
  const int64_t token = blockIdx.x;
  const float gate = 1.0f / (1.0f + __expf(-__bfloat162float(gate_logits[token])));

  for (int64_t c = threadIdx.x; c < width; c += blockDim.x) {
    const int64_t i = token * width + c;
    out[i] = __float2bfloat16(__bfloat162float(out[i]) +
                              gate * __bfloat162float(shared[i]));
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

Status MoeRouteTopK(const TensorView& logits, const TensorView& out_weights,
                    const TensorView& out_experts, bool renormalize,
                    float scale, cudaStream_t stream) {
  if (!(scale > 0.0f) || !isfinite(scale)) {
    return InvalidArgumentError("gate scale must be positive and finite, got ",
                                scale);
  }

  INFERX_RETURN_IF_ERROR(CheckTensor(logits, DataType::kBFloat16, 2, "logits"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(out_weights, DataType::kFloat, 2, "out_weights"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(out_experts, DataType::kInt32, 2, "out_experts"));

  const int64_t tokens = logits.Dim(0);
  const int64_t num_experts = logits.Dim(1);
  const int64_t k = out_weights.Dim(1);

  if (out_weights.Dim(0) != tokens || out_experts.Dim(0) != tokens ||
      out_experts.Dim(1) != k) {
    return InvalidArgumentError(
        "router outputs disagree: logits ", logits.GetShape().ToString(),
        ", weights ", out_weights.GetShape().ToString(), ", experts ",
        out_experts.GetShape().ToString());
  }

  if (k <= 0 || k > kMaxTopK) {
    return InvalidArgumentError("top-k must be in [1, ", kMaxTopK, "], got ", k);
  }

  if (k > num_experts) {
    return InvalidArgumentError("cannot route to ", k, " of ", num_experts,
                                " experts");
  }

  if (num_experts > kMaxExperts) {
    return InvalidArgumentError("num_experts ", num_experts, " exceeds the ",
                                kMaxExperts,
                                " one router block holds in shared memory");
  }

  if (tokens == 0) return OkStatus();

  const size_t shared_bytes = static_cast<size_t>(num_experts) * sizeof(float);

  RouteTopKKernel<<<static_cast<int>(tokens), kBlock, shared_bytes, stream>>>(
      static_cast<const bf16*>(logits.Data()),
      static_cast<float*>(out_weights.Data()),
      static_cast<int32_t*>(out_experts.Data()), num_experts,
      static_cast<int>(k), renormalize, scale);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status MoeBuildDispatch(const TensorView& experts, int64_t num_experts,
                        const TensorView& out_offsets, const TensorView& out_rows,
                        const TensorView& out_dest, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(experts, DataType::kInt32, 2, "experts"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(out_offsets, DataType::kInt32, 1, "out_offsets"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out_rows, DataType::kInt32, 1, "out_rows"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out_dest, DataType::kInt32, 1, "out_dest"));

  const int64_t tokens = experts.Dim(0);
  const int64_t k = experts.Dim(1);
  const int64_t assignments = tokens * k;

  if (num_experts <= 0) {
    return InvalidArgumentError("num_experts must be positive, got ",
                                num_experts);
  }

  if (out_offsets.Dim(0) != num_experts + 1) {
    return InvalidArgumentError("out_offsets must hold ", num_experts + 1,
                                " entries, got ", out_offsets.Dim(0));
  }

  if (out_rows.Dim(0) != assignments || out_dest.Dim(0) != assignments) {
    return InvalidArgumentError("out_rows/out_dest must hold ", assignments,
                                " entries, got ", out_rows.Dim(0), " and ",
                                out_dest.Dim(0));
  }

  if (assignments == 0) {
    // Still has to zero the offsets: a caller reading offsets[e+1]-offsets[e]
    // for an empty batch must see zero counts rather than uninitialized memory.
    INFERX_CUDA_RETURN_IF_ERROR(cudaMemsetAsync(
        out_offsets.Data(), 0, sizeof(int32_t) * (num_experts + 1), stream));
    return OkStatus();
  }

  // Three launches, no scratch and no synchronization: the counts are written
  // into the offsets buffer and scanned in place, so the whole dispatch is
  // stream-ordered device work and captures into a CUDA graph like the rest of
  // a decode step.
  const int grid = static_cast<int>(num_experts);
  auto* offsets_ptr = static_cast<int32_t*>(out_offsets.Data());

  CountPerExpertKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const int32_t*>(experts.Data()), assignments, offsets_ptr);
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  ScanOffsetsKernel<<<1, 1, 0, stream>>>(offsets_ptr, num_experts);
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  ScatterByExpertKernel<<<grid, kBlock, 0, stream>>>(
      static_cast<const int32_t*>(experts.Data()), assignments,
      static_cast<int>(k), offsets_ptr,
      static_cast<int32_t*>(out_rows.Data()),
      static_cast<int32_t*>(out_dest.Data()));
  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());

  return OkStatus();
}

Status MoeGatherRows(const TensorView& x, const TensorView& rows,
                     const TensorView& out, cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(x, DataType::kBFloat16, 2, "x"));
  INFERX_RETURN_IF_ERROR(CheckTensor(rows, DataType::kInt32, 1, "rows"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));

  const int64_t width = x.Dim(1);

  if (out.Dim(1) != width) {
    return InvalidArgumentError("x is ", width, " wide but out is ",
                                out.Dim(1));
  }

  if (out.Dim(0) != rows.Dim(0)) {
    return InvalidArgumentError("rows has ", rows.Dim(0), " entries but out "
                                "has ", out.Dim(0), " rows");
  }

  if (rows.Dim(0) == 0) return OkStatus();

  GatherRowsKernel<<<static_cast<int>(rows.Dim(0)), kBlock, 0, stream>>>(
      static_cast<const bf16*>(x.Data()),
      static_cast<const int32_t*>(rows.Data()),
      static_cast<bf16*>(out.Data()), width);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status MoeCombineRows(const TensorView& y, const TensorView& dest,
                      const TensorView& weights, const TensorView& out,
                      cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(y, DataType::kBFloat16, 2, "y"));
  INFERX_RETURN_IF_ERROR(CheckTensor(dest, DataType::kInt32, 1, "dest"));
  INFERX_RETURN_IF_ERROR(CheckTensor(weights, DataType::kFloat, 2, "weights"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));

  const int64_t tokens = out.Dim(0);
  const int64_t width = out.Dim(1);
  const int64_t k = weights.Dim(1);

  if (weights.Dim(0) != tokens) {
    return InvalidArgumentError("weights has ", weights.Dim(0), " rows but "
                                "out has ", tokens);
  }

  if (dest.Dim(0) != tokens * k) {
    return InvalidArgumentError("dest must hold ", tokens * k, " entries, got ",
                                dest.Dim(0));
  }

  if (y.Dim(1) != width) {
    return InvalidArgumentError("y is ", y.Dim(1), " wide but out is ", width);
  }

  if (tokens == 0) return OkStatus();

  CombineRowsKernel<<<static_cast<int>(tokens), kBlock, 0, stream>>>(
      static_cast<const bf16*>(y.Data()),
      static_cast<const int32_t*>(dest.Data()),
      static_cast<const float*>(weights.Data()),
      static_cast<bf16*>(out.Data()), width, static_cast<int>(k));

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

Status MoeAddSharedExpert(const TensorView& shared,
                          const TensorView& gate_logits, const TensorView& out,
                          cudaStream_t stream) {
  INFERX_RETURN_IF_ERROR(CheckTensor(shared, DataType::kBFloat16, 2, "shared"));
  INFERX_RETURN_IF_ERROR(
      CheckTensor(gate_logits, DataType::kBFloat16, 2, "gate_logits"));
  INFERX_RETURN_IF_ERROR(CheckTensor(out, DataType::kBFloat16, 2, "out"));

  const int64_t tokens = out.Dim(0);
  const int64_t width = out.Dim(1);

  if (shared.Dim(0) != tokens || shared.Dim(1) != width) {
    return InvalidArgumentError("shared is ", shared.GetShape().ToString(),
                                " but out is ", out.GetShape().ToString());
  }

  if (gate_logits.Dim(0) != tokens || gate_logits.Dim(1) != 1) {
    return InvalidArgumentError("gate_logits must be [", tokens, ", 1], got ",
                                gate_logits.GetShape().ToString());
  }

  if (tokens == 0) return OkStatus();

  AddSharedExpertKernel<<<static_cast<int>(tokens), kBlock, 0, stream>>>(
      static_cast<const bf16*>(shared.Data()),
      static_cast<const bf16*>(gate_logits.Data()),
      static_cast<bf16*>(out.Data()), width);

  INFERX_CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return OkStatus();
}

}  // namespace inferx::kernels
