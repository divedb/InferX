// single_prefill_demo.cu
//
// Standalone demo of FlashInfer's SinglePrefillWithKVCacheDispatched (FA2 path).
//
// Computes scaled-dot-product attention for a single request:
//
//   O = softmax(Q @ K^T * scale) @ V
//
// with optional causal masking and GQA (grouped-query attention) support.
// The GPU result is verified against a naive CPU reference implementation.
//
// This demonstrates the lowest-level C++ entry point into FlashInfer's FA2
// prefill kernel -- the same code path that Python's
// `flashinfer.single_prefill_with_kv_cache(backend="fa2")` ultimately calls,
// minus the TVM-FFI / PyTorch binding layer.

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>

using half_t = half;

// ---------------------------------------------------------------------------
// Helper: allocate a device buffer and copy from host.
// ---------------------------------------------------------------------------
template <typename T>
T* to_device(const std::vector<T>& h_vec, cudaStream_t stream) {
  T* d_ptr = nullptr;
  if (cudaMalloc(&d_ptr, h_vec.size() * sizeof(T)) != cudaSuccess) {
    std::fprintf(stderr, "cudaMalloc failed for %zu bytes\n", h_vec.size() * sizeof(T));
    std::exit(1);
  }
  cudaMemcpyAsync(d_ptr, h_vec.data(), h_vec.size() * sizeof(T),
                  cudaMemcpyHostToDevice, stream);
  return d_ptr;
}

// ---------------------------------------------------------------------------
// CPU: apply Llama-style RoPE to a single head vector in-place.
//
// Splits head_dim into first/second halves and rotates each pair
// (x[d], x[d + D/2]) by angle = pos * theta^(-2d/D) / rope_scale.
// This matches FlashInfer's q_frag_apply_llama_rope / k_frag_apply_llama_rope
// (prefill.cuh:202, 221) exactly.
// ---------------------------------------------------------------------------
void cpu_apply_rope(half_t* vec, int pos, int head_dim,
                    float rope_theta, float rope_scale) {
  const int half_dim = head_dim / 2;
  for (int d = 0; d < half_dim; ++d) {
    float freq =
        1.f / powf(rope_theta, static_cast<float>(2 * d) / head_dim) / rope_scale;
    float angle = pos * freq;
    float c, s;
    sincosf(angle, &s, &c);
    float x0 = __half2float(vec[d]);
    float x1 = __half2float(vec[d + half_dim]);
    vec[d]          = __float2half(x0 * c - x1 * s);
    vec[d + half_dim] = __float2half(x0 * s + x1 * c);
  }
}

// ---------------------------------------------------------------------------
// CPU reference: naive O(N^2 * D) attention with optional RoPE.
//
// Q layout: [qo_len, num_qo_heads, head_dim]  (NHD, row-major)
// K layout: [kv_len, num_kv_heads, head_dim]
// V layout: [kv_len, num_kv_heads, head_dim]
// O layout: [qo_len, num_qo_heads, head_dim]
//
// Position convention (matches FlashInfer's SinglePrefillWithKVCacheDevice):
//   Q token i  -> absolute position (kv_len - qo_len + i)
//   K token j  -> absolute position j
// ---------------------------------------------------------------------------
void cpu_attention(std::vector<half_t> q,           // by value: RoPE mutates in-place
                   std::vector<half_t> k,           // by value
                   const std::vector<half_t>& v,
                   std::vector<half_t>& o,
                   int qo_len, int kv_len,
                   int num_qo_heads, int num_kv_heads, int head_dim,
                   bool causal, float scale,
                   bool use_rope, float rope_theta, float rope_scale) {
  const int group_size = num_qo_heads / num_kv_heads;
  const int offset = kv_len - qo_len;  // >= 0 for valid causal

  // -- Pre-apply RoPE to Q and K (if enabled) -------------------------------
  // Q position: kv_len - qo_len + i  (query tokens are the last qo_len tokens)
  // K position: j                    (absolute KV token index)
  if (use_rope) {
    for (int i = 0; i < qo_len; ++i) {
      const int q_pos = kv_len - qo_len + i;
      for (int h = 0; h < num_qo_heads; ++h) {
        cpu_apply_rope(&q[(static_cast<size_t>(i) * num_qo_heads + h) * head_dim],
                       q_pos, head_dim, rope_theta, rope_scale);
      }
    }
    for (int j = 0; j < kv_len; ++j) {
      for (int kvh = 0; kvh < num_kv_heads; ++kvh) {
        cpu_apply_rope(&k[(static_cast<size_t>(j) * num_kv_heads + kvh) * head_dim],
                       j, head_dim, rope_theta, rope_scale);
      }
    }
  }

  std::vector<float> scores(kv_len);

  for (int h = 0; h < num_qo_heads; ++h) {
    const int kvh = h / group_size;
    for (int i = 0; i < qo_len; ++i) {
      // Q @ K^T * scale
      float max_score = -1e30f;
      for (int j = 0; j < kv_len; ++j) {
        if (causal && (j - offset > i)) {
          scores[j] = -1e30f;
          continue;
        }
        float dot = 0.f;
        const half_t* q_row =
            &q[(static_cast<size_t>(i) * num_qo_heads + h) * head_dim];
        const half_t* k_row =
            &k[(static_cast<size_t>(j) * num_kv_heads + kvh) * head_dim];
        for (int d = 0; d < head_dim; ++d) {
          dot += __half2float(q_row[d]) * __half2float(k_row[d]);
        }
        scores[j] = dot * scale;
        if (scores[j] > max_score) max_score = scores[j];
      }

      // softmax
      float sum = 0.f;
      for (int j = 0; j < kv_len; ++j) {
        scores[j] = expf(scores[j] - max_score);
        sum += scores[j];
      }

      // weighted sum of V
      half_t* o_row =
          &o[(static_cast<size_t>(i) * num_qo_heads + h) * head_dim];
      for (int d = 0; d < head_dim; ++d) {
        float acc = 0.f;
        for (int j = 0; j < kv_len; ++j) {
          const half_t* v_row =
              &v[(static_cast<size_t>(j) * num_kv_heads + kvh) * head_dim];
          acc += scores[j] * __half2float(v_row[d]);
        }
        o_row[d] = __float2half(acc / sum);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Run one FlashInfer single-prefill case and compare against CPU.
// ---------------------------------------------------------------------------
bool run_case(int qo_len, int kv_len,
              int num_qo_heads, int num_kv_heads,
              int head_dim, bool causal,
              bool use_rope = true, float rope_theta = 10000.f,
              float rope_scale = 1.f) {
  const float scale = 1.f / sqrtf(static_cast<float>(head_dim));

  const size_t q_elems = static_cast<size_t>(qo_len) * num_qo_heads * head_dim;
  const size_t kv_elems = static_cast<size_t>(kv_len) * num_kv_heads * head_dim;

  // -- Random init (deterministic seed) -------------------------------------
  srand(42);
  auto rand_half_vec = [](size_t n, int seed) {
    srand(seed);
    std::vector<half_t> v(n);
    for (size_t i = 0; i < n; ++i) {
      // range [-1, 1]
      float r = static_cast<float>(rand()) / RAND_MAX * 2.f - 1.f;
      v[i] = __float2half(r);
    }
    return v;
  };

  std::vector<half_t> h_q = rand_half_vec(q_elems, 42);
  std::vector<half_t> h_k = rand_half_vec(kv_elems, 43);
  std::vector<half_t> h_v = rand_half_vec(kv_elems, 44);
  std::vector<half_t> h_o(q_elems, __float2half(0.f));

  // -- CPU reference --------------------------------------------------------
  cpu_attention(h_q, h_k, h_v, h_o, qo_len, kv_len,
                num_qo_heads, num_kv_heads, head_dim, causal, scale,
                use_rope, rope_theta, rope_scale);

  // -- GPU setup ------------------------------------------------------------
  cudaStream_t stream;
  cudaStreamCreate(&stream);

  half_t* d_q = to_device(h_q, stream);
  half_t* d_k = to_device(h_k, stream);
  half_t* d_v = to_device(h_v, stream);
  half_t* d_o = to_device(h_o, stream);  // will be overwritten

  // -- Build Params ---------------------------------------------------------
  // SinglePrefillParams is the framework-agnostic parameter struct defined in
  // include/flashinfer/attention/default_prefill_params.cuh. It carries raw
  // pointers + strides, no torch/tvm dependency.
  using Params = flashinfer::SinglePrefillParams<half_t, half_t, half_t>;

  Params params;
  params.q = d_q;
  params.k = d_k;
  params.v = d_v;
  params.o = d_o;
  params.lse = nullptr;
  params.maybe_custom_mask = nullptr;
  params.maybe_alibi_slopes = nullptr;
  params.num_qo_heads = num_qo_heads;
  params.num_kv_heads = num_kv_heads;
  params.group_size = flashinfer::uint_fastdiv(num_qo_heads / num_kv_heads);
  params.qo_len = qo_len;
  params.kv_len = kv_len;
  // NHD layout: q[qo_len, num_qo_heads, head_dim] -- row stride is the full
  // head-contiguous row, head stride is head_dim.
  params.q_stride_n = num_qo_heads * head_dim;
  params.q_stride_h = head_dim;
  params.k_stride_n = num_kv_heads * head_dim;
  params.k_stride_h = head_dim;
  params.v_stride_n = num_kv_heads * head_dim;
  params.v_stride_h = head_dim;
  params.head_dim = head_dim;
  params.window_left = -1;          // no sliding window
  params.logits_soft_cap = 0.f;     // no soft cap
  params.sm_scale = scale;
  params.rope_rcp_scale = 1.f / rope_scale;  // reciprocal, as FlashInfer expects
  params.rope_rcp_theta = 1.f / rope_theta;
  params.partition_kv = false;

  // -- Dispatch -------------------------------------------------------------
  // Template parameters fixed for this demo. A production wrapper dispatches
  // over HEAD_DIM / POS_ENCODING_MODE / MASK_MODE at runtime; here we compile
  // one specialization per (head_dim, causal) pair.
  //
  // DefaultAttention<use_custom_mask, use_sliding_window,
  //                  use_logits_soft_cap, use_alibi>  -- all off for the
  // vanilla softmax path.
  using Variant = flashinfer::DefaultAttention<false, false, false, false>;

  constexpr uint32_t HEAD_DIM_QK = 128;
  constexpr uint32_t HEAD_DIM_VO = 128;

  // tmp == nullptr tells the dispatcher "do not split-KV" -- enough parallelism
  // comes from num_kv_heads * ceil(qo_len*group_size / CTA_TILE_Q).
  half_t* tmp = nullptr;

  // Select PosEncodingMode at compile time. FlashInfer's dispatch is a template,
  // so kNone and kRoPELlama are separate instantiations. This branch picks one.
  using PE = flashinfer::PosEncodingMode;
  constexpr PE kPosMode = flashinfer::PosEncodingMode::kRoPELlama;

  cudaError_t err;
  if (causal) {
    err = flashinfer::SinglePrefillWithKVCacheDispatched<
        HEAD_DIM_QK, HEAD_DIM_VO, kPosMode,
        /*USE_FP16_QK_REDUCTION=*/false,
        flashinfer::MaskMode::kCausal, Variant>(
        params, tmp, stream);
  } else {
    err = flashinfer::SinglePrefillWithKVCacheDispatched<
        HEAD_DIM_QK, HEAD_DIM_VO, kPosMode,
        /*USE_FP16_QK_REDUCTION=*/false,
        flashinfer::MaskMode::kNone, Variant>(
        params, tmp, stream);
  }

  if (err != cudaSuccess) {
    std::fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(err));
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
    cudaStreamDestroy(stream);
    return false;
  }

  // -- Copy back & compare --------------------------------------------------
  std::vector<half_t> h_o_gpu(q_elems, __float2half(0.f));
  cudaMemcpyAsync(h_o_gpu.data(), d_o, q_elems * sizeof(half_t),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  float max_abs_err = 0.f;
  float max_rel_err = 0.f;
  for (size_t i = 0; i < q_elems; ++i) {
    float ref = __half2float(h_o[i]);
    float gpu = __half2float(h_o_gpu[i]);
    float abs_err = fabsf(gpu - ref);
    float rel_err = (fabsf(ref) > 1e-4f) ? abs_err / fabsf(ref) : 0.f;
    if (abs_err > max_abs_err) max_abs_err = abs_err;
    if (rel_err > max_rel_err) max_rel_err = rel_err;
  }

  // Cleanup
  cudaFree(d_q); cudaFree(d_k); cudaFree(d_v); cudaFree(d_o);
  cudaStreamDestroy(stream);

  const char* mask_str = causal ? "causal" : "none  ";
  const char* rope_str = use_rope ? "rope" : "norope";
  bool ok = max_abs_err < 1e-2f;
  std::printf("  [%s] qo=%4d  kv=%4d  heads=%2d/%d  dim=%d  mask=%s  %s  "
              "max_abs=%.2e  max_rel=%.2e  %s\n",
              ok ? "OK" : "FAIL",
              qo_len, kv_len, num_qo_heads, num_kv_heads, head_dim,
              mask_str, rope_str, max_abs_err, max_rel_err,
              ok ? "" : "(threshold exceeded)");
  return ok;
}

// ---------------------------------------------------------------------------
int main() {
  // Device check: FA2 prefill requires sm_80+ for bf16, fp16 works on sm_75+.
  int device_count = 0;
  cudaError_t device_err = cudaGetDeviceCount(&device_count);
  if (device_err != cudaSuccess || device_count == 0) {
    std::fprintf(stderr, "single_prefill_demo: no CUDA device available%s%s\n",
                 device_err == cudaSuccess ? "" : ": ",
                 device_err == cudaSuccess ? "" : cudaGetErrorString(device_err));
    return 77;
  }

  int dev_id = 0;
  device_err = cudaGetDevice(&dev_id);
  if (device_err != cudaSuccess) {
    std::fprintf(stderr, "single_prefill_demo: cudaGetDevice failed: %s\n",
                 cudaGetErrorString(device_err));
    return 1;
  }
  cudaDeviceProp prop;
  device_err = cudaGetDeviceProperties(&prop, dev_id);
  if (device_err != cudaSuccess) {
    std::fprintf(stderr,
                 "single_prefill_demo: cudaGetDeviceProperties failed: %s\n",
                 cudaGetErrorString(device_err));
    return 1;
  }
  std::printf("FlashInfer SinglePrefill demo\n");
  std::printf("  device : %s (sm_%d%d)\n", prop.name,
              prop.major, prop.minor);
  std::printf("  dtype  : fp16 (half)\n");
  std::printf("  backend: FA2 (non-Hopper path)\n");
  std::printf("  RoPE   : Llama-style, theta=10000\n\n");

  constexpr int HEAD_DIM = 128;

  // All cases now use RoPE (theta=10000, scale=1.0) -- the Llama 2 default.
  // The dispatch picks CTA_TILE_Q / NUM_MMA_Q / NUM_MMA_KV automatically inside
  // the template.
  bool all_ok = true;

  std::printf("Running test cases (head_dim=%d, rope=on):\n", HEAD_DIM);

  // MHA, no causal, short sequence (CTA_TILE_Q=64 territory)
  all_ok &= run_case(37,  501, 4, 4, HEAD_DIM, /*causal=*/false);
  all_ok &= run_case(37,  501, 4, 4, HEAD_DIM, /*causal=*/true);

  // MHA, longer sequence (CTA_TILE_Q=128 territory)
  all_ok &= run_case(1024, 1024, 8, 8, HEAD_DIM, /*causal=*/false);
  all_ok &= run_case(1024, 1024, 8, 8, HEAD_DIM, /*causal=*/true);

  // GQA: 4 Q heads, 1 KV head (group_size=4)
  all_ok &= run_case(577, 2042, 4, 1, HEAD_DIM, /*causal=*/false);
  all_ok &= run_case(577, 2042, 4, 1, HEAD_DIM, /*causal=*/true);

  // GQA: 7 Q heads, 1 KV head (group_size=7, non-power-of-2)
  all_ok &= run_case(127,  501, 7, 1, HEAD_DIM, /*causal=*/false);

  // kv_len > qo_len (prefill into existing cache) -- exercises non-zero Q
  // position offset (kv_len - qo_len) in RoPE
  all_ok &= run_case(128, 4096, 8, 2, HEAD_DIM, /*causal=*/true);

  // RoPE with position interpolation (rope_scale=2.0, doubles the effective
  // context length at the cost of resolution -- "PI" scaling).
  all_ok &= run_case(512, 512, 8, 8, HEAD_DIM, /*causal=*/true,
                     /*rope_theta=*/10000.f, /*rope_scale=*/2.f);

  std::printf("\n%s\n", all_ok
      ? "All cases passed (max_abs < 1e-2)."
      : "Some cases FAILED -- check kernel parameters.");

  return all_ok ? 0 : 1;
}
