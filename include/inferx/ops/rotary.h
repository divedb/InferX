#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Which frequency-scaling flavor a rotary embedding uses.
enum class RopeFlavor : uint8_t {
  kDefault,  ///< Unscaled theta^(-2i/dim).
  kLinear,   ///< All frequencies divided by factor (Position Interpolation).
  kLlama3,   ///< Wavelength-blended interpolation (Llama 3.1).
  kYarn,     ///< Ramped interpolation plus a softmax temperature (YaRN).
};

/// \brief Frozen scaling parameters of one rotary embedding.
///
/// Raw checkpoint fields are finalized into these by the family translator:
/// YaRN's effective factor and ramp bounds and the attention temperature are
/// computed once on the host, so kernels and validation share one formula.
struct RopeScaling {
  RopeFlavor flavor = RopeFlavor::kDefault;
  float factor = 1.0f;                ///< Interpolation factor; >1 extends context.
  float low_freq_factor = 1.0f;       ///< Llama3: wavelengths above this interpolate fully.
  float high_freq_factor = 4.0f;      ///< Llama3: wavelengths below this stay extrapolated.
  float original_max_position = 0.0f; ///< Llama3 old context; YaRN original length.
  float ramp_low = 0.0f;              ///< YaRN ramp start, in frequency indices.
  float ramp_high = 0.0f;             ///< YaRN ramp end, in frequency indices.
  float attention_scale = 1.0f;       ///< YaRN temperature folded into cos/sin.
};

/// \brief Parameters of one rotary position encoding.
struct RotaryParams {
  int64_t rotary_dim = 0;  ///< Rotated columns per head; even, <= head_dim.
  float theta = 10000.0f;  ///< Base frequency.
  RopeScaling scaling;
};

/// \brief Inverse frequency of complex pair `i` in [0, rotary_dim/2).
///
/// One formula serves every flavor, the CUDA kernels, host validation, and
/// tests; it matches the reference implementations of each scaling method
/// (HF transformers rope init and vLLM rotary_embedding). Uses the C math
/// intrinsics so it compiles for both host and device.
#ifdef __CUDACC__
#define INFERX_ROPE_HD __host__ __device__
#else
#define INFERX_ROPE_HD inline
#endif
INFERX_ROPE_HD float InverseFrequency(int64_t i, const RotaryParams& p) {
  const float exponent =
      2.0f * static_cast<float>(i) / static_cast<float>(p.rotary_dim);
  const float inv = ::powf(p.theta, -exponent);
  const RopeScaling& s = p.scaling;
  switch (s.flavor) {
    case RopeFlavor::kLinear:
      return inv / s.factor;
    case RopeFlavor::kLlama3: {
      const float wavelen = 6.28318530717958647692f / inv;
      const float low = s.original_max_position / s.low_freq_factor;
      const float high = s.original_max_position / s.high_freq_factor;
      if (wavelen > low) return inv / s.factor;
      if (wavelen < high) return inv;
      const float t = (s.original_max_position / wavelen - s.low_freq_factor) /
                      (s.high_freq_factor - s.low_freq_factor);
      return (1.0f - t) * inv / s.factor + t * inv;
    }
    case RopeFlavor::kYarn: {
      const float span = s.ramp_high - s.ramp_low;
      const float ramp =
          span > 0.0f
              ? ::fminf(::fmaxf((static_cast<float>(i) - s.ramp_low) / span, 0.0f), 1.0f)
              : 1.0f;
      return (inv / s.factor) * ramp + inv * (1.0f - ramp);
    }
    default:
      return inv;
  }
}
#undef INFERX_ROPE_HD

/// \brief Applies neox-style (rotate-half) RoPE to `q` and `k`, in place.
///
/// For each head, the first and second halves of the rotary window form
/// complex pairs: x[i] and x[i + rotary_dim/2] rotate by
/// position * theta^(-2i/rotary_dim). Columns past `rotary_dim` pass through
/// untouched. Frequencies and trigonometry run in fp32; cosine/sine values
/// and each multiply/add are rounded to the activation dtype, matching the
/// reference BF16 cache and rotation. Work is enqueued on the context's stream.
///
/// \param ctx        Execution context; all tensors must live on ctx.device().
/// \param q          [tokens, query_heads, head_dim] queries; updated in place.
/// \param k          [tokens, kv_heads, head_dim] keys; updated in place.
/// \param positions  [tokens] int32 position of each token.
/// \param params     Rotary parameters.
/// \return           OK, or InvalidArgument/Unimplemented for bad inputs.
Status ApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                 const Tensor& positions, const RotaryParams& params);

/// Rounded per-head RMSNorm followed by RoPE, fused for BF16 heads of width
/// 128. Preserves the intermediate BF16 normalization and rotation roundings.
Status NormalizeAndApplyRope(ExecutionContext& ctx, const Tensor& q, const Tensor& k,
                             const Tensor& q_weight, const Tensor& k_weight,
                             const Tensor& positions, float eps, const RotaryParams& params);

}  // namespace inferx::ops
