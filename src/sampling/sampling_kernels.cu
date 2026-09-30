// Sampling pipeline kernels: bias, allowlist, penalties, temperature, the
// top-k/top-p/min-p filters, and a counter-based draw. One block per row;
// greedy rows degenerate to the argmax tie break the fused greedy op uses.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>
#include <cmath>

#include "inferx/sampling/sampling_kernels.h"

namespace inferx::sampling::cuda {
namespace {

constexpr int kThreads = 256;

/// Counter-based RNG: splitmix64 over (seed, stream offset, row). Draws
/// depend only on (seed, generated index), never on batch composition.
__device__ float NextUnit(uint64_t& state) {
  state += 0x9E3779B97F4A7C15ull;
  uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  z = z ^ (z >> 31);
  return static_cast<float>(z >> 40) * (1.0f / 16777216.0f);  // 24 bits.
}

struct Candidate {
  float value;
  int index;
};

__device__ Candidate Better(Candidate a, Candidate b) {
  return b.value > a.value || (b.value == a.value && b.index < a.index) ? b : a;
}

/// Block-wide reduction whose result every thread receives (the naive
/// shuffle-down tail leaves non-zero lanes with partials).
__device__ Candidate ReduceCandidate(Candidate a) {
  __shared__ float values[8];
  __shared__ int indices[8];
  __shared__ float broadcast_value;
  __shared__ int broadcast_index;
  const int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int delta = 16; delta; delta /= 2) {
    const Candidate b{__shfl_down_sync(0xffffffffu, a.value, delta),
                      __shfl_down_sync(0xffffffffu, a.index, delta)};
    a = Better(a, b);
  }
  if (lane == 0) {
    values[warp] = a.value;
    indices[warp] = a.index;
  }
  __syncthreads();
  a = lane < 8 ? Candidate{values[lane], indices[lane]} : Candidate{-INFINITY, INT_MAX};
  if (warp == 0) {
    for (int delta = 16; delta; delta /= 2) {
      const Candidate b{__shfl_down_sync(0xffffffffu, a.value, delta),
                        __shfl_down_sync(0xffffffffu, a.index, delta)};
      a = Better(a, b);
    }
    if (lane == 0) {
      broadcast_value = a.value;
      broadcast_index = a.index;
    }
  }
  __syncthreads();
  return Candidate{broadcast_value, broadcast_index};
}

__device__ float BlockSum(float partial) {
  __shared__ float sums[kThreads / 32];
  for (int delta = 16; delta; delta /= 2) partial += __shfl_down_sync(~0u, partial, delta);
  if (threadIdx.x % 32 == 0) sums[threadIdx.x / 32] = partial;
  __syncthreads();
  __shared__ float total;
  if (threadIdx.x == 0) {
    float t = 0;
    for (int w = 0; w < kThreads / 32; ++w) t += sums[w];
    total = t;
  }
  __syncthreads();
  return total;
}

/// \brief One block per row of the logits matrix.
///
/// The filters select by repeated max extraction: selected probabilities
/// are negated (magnitude preserved), which both removes them from later
/// rounds and records the selected set. top-k takes k rounds; top-p stops
/// once the cumulative mass reaches the threshold (the crossing token is
/// kept, matching HF semantics); min-p drops tokens below a fraction of
/// the maximum probability before the others.
__global__ void SampleRowsKernel(const __nv_bfloat16* __restrict__ logits,
                                 const float* __restrict__ temperature,
                                 const int* __restrict__ top_k,
                                 const float* __restrict__ top_p,
                                 const float* __restrict__ min_p,
                                 const float* __restrict__ penalties,
                                 const unsigned long long* __restrict__ seeds,
                                 const unsigned long long* __restrict__ rng_offsets,
                                 const int* __restrict__ greedy, const int* __restrict__ bias_ptr,
                                 const unsigned long long* __restrict__ bias_entries,
                                 const int* __restrict__ allow_ptr,
                                 const int* __restrict__ allow_entries,
                                 const int* __restrict__ hist_ptr,
                                 const unsigned long long* __restrict__ hist_entries, int vocab,
                                 float* __restrict__ probs, int* __restrict__ out) {
  const int row = blockIdx.x;
  float* p = probs + int64_t(row) * vocab;
  const int tid = threadIdx.x;

  // Stage the row from logits, add sparse bias, then hard-mask to the
  // allowlist; banned tokens end at -INFINITY.
  for (int j = tid; j < vocab; j += blockDim.x) {
    p[j] = __bfloat162float(logits[int64_t(row) * vocab + j]);
  }
  __syncthreads();
  for (int e = bias_ptr[row] + tid; e < bias_ptr[row + 1]; e += blockDim.x) {
    const uint64_t packed = bias_entries[e];
    const int token = static_cast<int>(packed >> 32);
    const uint32_t bits = static_cast<uint32_t>(packed & 0xFFFFFFFFull);
    const float bias = __int_as_float(static_cast<int>(bits));
    if (p[token] != -INFINITY) atomicAdd(&p[token], bias);
  }
  __syncthreads();
  const bool has_allow = allow_ptr[row + 1] > allow_ptr[row];
  if (has_allow) {
    for (int j = tid; j < vocab; j += blockDim.x) {
      bool ok = false;
      for (int e = allow_ptr[row]; e < allow_ptr[row + 1]; ++e) {
        if (allow_entries[e] == j) {
          ok = true;
          break;
        }
      }
      if (!ok) p[j] = -INFINITY;
    }
    __syncthreads();
  }

  // Penalties over the (deduplicated, counted) history CSR.
  const float rep = penalties[row * 3], pres = penalties[row * 3 + 1],
              freq = penalties[row * 3 + 2];
  if (rep != 1.0f || pres != 0.0f || freq != 0.0f) {
    for (int e = hist_ptr[row] + tid; e < hist_ptr[row + 1]; e += blockDim.x) {
      const uint64_t packed = hist_entries[e];
      const int token = static_cast<int>(packed >> 32);
      const int count = static_cast<int>(packed & 0xFFFFFFFFull);
      float v = p[token];
      if (rep != 1.0f) v = v > 0 ? v / rep : v * rep;
      v -= pres + freq * count;
      p[token] = v;
    }
    __syncthreads();
  }

  // Softmax over the (possibly biased/masked/penalized) row.
  Candidate m{-INFINITY, INT_MAX};
  for (int j = tid; j < vocab; j += blockDim.x) m = Better(m, {p[j], j});
  m = ReduceCandidate(m);
  const float max_value = m.value;
  float partial = 0;
  for (int j = tid; j < vocab; j += blockDim.x) {
    p[j] = p[j] == -INFINITY ? 0.0f : __expf(p[j] - max_value);
    partial += p[j];
  }
  const float total = BlockSum(partial);
  __syncthreads();

  if (greedy[row]) {
    if (tid == 0) out[row] = total == 0 ? 0 : m.index;
    return;
  }

  // Temperature (after softmax inputs are set, rescale logits equivalent:
  // dividing logits by t equals raising to 1/t in probability, applied by
  // rebuilding from the stored row).
  const float t = temperature[row];
  if (t != 1.0f && t > 0.0f) {
    const float inv = 1.0f / t;
    for (int j = tid; j < vocab; j += blockDim.x) {
      // p was built with max subtraction: log p = logit - max.
      p[j] = p[j] > 0 ? __powf(p[j], inv) : 0.0f;
    }
    __syncthreads();
  }

  // min-p: drop below the fraction of the (current) maximum.
  if (min_p[row] > 0.0f) {
    Candidate c{-1.0f, INT_MAX};
    for (int j = tid; j < vocab; j += blockDim.x) c = Better(c, {p[j], j});
    c = ReduceCandidate(c);
    const float threshold = min_p[row] * c.value;
    for (int j = tid; j < vocab; j += blockDim.x) {
      if (p[j] < threshold) p[j] = 0.0f;
    }
    __syncthreads();
  }

  // top-k / top-p by repeated max extraction; negation marks selection.
  const int k = top_k[row];
  const float p_cut = top_p[row];
  const bool filter_k = k > 0 && k < vocab;
  const bool filter_p = p_cut < 1.0f;
  if (filter_k || filter_p) {
    // The nucleus threshold is a share of the current (post-temperature,
    // post-min-p) mass, so measure it against the live sum.
    float mass = 0;
    for (int j = tid; j < vocab; j += blockDim.x) mass += p[j];
    mass = BlockSum(mass);
    float cumulative = 0;
    const int rounds = filter_k ? k : vocab;
    for (int r = 0; r < rounds; ++r) {
      Candidate c{0.0f, INT_MAX};  // Only non-negative probabilities remain.
      for (int j = tid; j < vocab; j += blockDim.x) c = Better(c, {p[j], j});
      c = ReduceCandidate(c);
      if (c.index == INT_MAX || c.value <= 0) break;
      cumulative += c.value;
      p[c.index] = -c.value;  // Selected: negate to exclude and remember.
      if (filter_p && cumulative >= p_cut * mass) break;
    }
    __syncthreads();
    for (int j = tid; j < vocab; j += blockDim.x) p[j] = p[j] < 0 ? -p[j] : 0.0f;
    __syncthreads();
  }

  // Renormalize and draw by inverse CDF (single-thread scan: vocab-sized,
  // correctness-first; the filters above did the heavy lifting in parallel).
  partial = 0;
  for (int j = tid; j < vocab; j += blockDim.x) partial += p[j];
  const float norm = BlockSum(partial);
  if (tid != 0) return;
  if (norm == 0) {
    out[row] = 0;
    return;
  }
  uint64_t state = seeds[row] ^ (rng_offsets[row] * 0x9E3779B97F4A7C15ull);
  const float u = NextUnit(state) * norm;
  float cdf = 0;
  int chosen = vocab - 1;
  for (int j = 0; j < vocab; ++j) {
    cdf += p[j];
    if (u < cdf) {
      chosen = j;
      break;
    }
  }
  out[row] = chosen;
}

Status CudaError(cudaError_t err, const char* what) {
  if (err != cudaSuccess) {
    return InternalError(what, " failed: ", cudaGetErrorString(err));
  }
  return OkStatus();
}

}  // namespace

Status SampleRows(ops::OpContext& ctx, const Tensor& logits, const DeviceParams& params,
                  const Tensor& probs_workspace, Tensor& output) {
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  const int batch = static_cast<int>(logits.Dim(0));
  const int vocab = static_cast<int>(logits.Dim(1));
  if (logits.GetDataType() != DataType::kBFloat16) {
    return UnimplementedError("sampling pipeline expects bfloat16 logits");
  }
  SampleRowsKernel<<<batch, kThreads, 0, static_cast<cudaStream_t>(ctx.GetStream())>>>(
      static_cast<const __nv_bfloat16*>(logits.Data()),
      static_cast<const float*>(params.temperature->Data()),
      static_cast<const int*>(params.top_k->Data()),
      static_cast<const float*>(params.top_p->Data()),
      static_cast<const float*>(params.min_p->Data()),
      static_cast<const float*>(params.penalties->Data()),
      static_cast<const unsigned long long*>(params.seeds->Data()),
      static_cast<const unsigned long long*>(params.rng_offsets->Data()),
      static_cast<const int*>(params.greedy->Data()),
      static_cast<const int*>(params.bias_ptr->Data()),
      static_cast<const unsigned long long*>(params.bias_entries->Data()),
      static_cast<const int*>(params.allow_ptr->Data()),
      static_cast<const int*>(params.allow_entries->Data()),
      static_cast<const int*>(params.hist_ptr->Data()),
      static_cast<const unsigned long long*>(params.hist_entries->Data()), vocab,
      static_cast<float*>(probs_workspace.Data()), static_cast<int*>(output.Data()));
  return CudaError(cudaGetLastError(), "sample rows launch");
}

}  // namespace inferx::sampling::cuda
