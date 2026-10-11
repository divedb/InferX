#ifndef INFERX_LAYERS_RMS_NORM_H_
#define INFERX_LAYERS_RMS_NORM_H_

#include <cstdint>

#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/norm.h"
#include "inferx/ops/op_context.h"

namespace inferx {
namespace models {

/// \brief RMS normalization module: a [hidden] scale plus its epsilon.
///
/// `Forward` normalizes in place-safe fashion (`out` may alias `x`);
/// `AddForward` is the fused residual form used between attention and MLP:
/// it stores the dtype-rounded `x + residual` back into `residual` and
/// writes the normalized sum to `out`. The rounding split is the kernel's
/// (FlashInfer keeps the fp32 sum on CUDA); both preserve reference
/// semantics.
class RMSNorm {
 public:
  RMSNorm() = default;

  /// \brief Allocates the [hidden] scale on `device`.
  ///
  /// \param round_before_weight True selects the Llama/Qwen reference
  ///        rounding (normalized activations round to the input dtype
  ///        before the scale multiply; the fused-add form normalizes the
  ///        dtype-rounded residual sum). False keeps the fused
  ///        single-rounding operator (Gemma lineage).
  static StatusOr<RMSNorm> Create(int64_t hidden_size, float eps, DataType dtype,
                                  Device device, bool round_before_weight = false);

  /// \brief out = x / sqrt(mean(x^2) + eps) * weight.
  Status Forward(ops::OpContext& ctx, const Tensor& x, Tensor& out) const;

  /// \brief residual = x + residual; out = RMSNorm(residual).
  ///
  /// `out` must be distinct from `residual` (it may alias `x`).
  Status AddForward(ops::OpContext& ctx, const Tensor& x, Tensor& residual,
                    Tensor& out) const;

  const Tensor& Weight() const { return weight_; }
  Tensor* MutableWeight() { return &weight_; }
  float Eps() const { return config_.eps; }
  int64_t HiddenSize() const { return hidden_size_; }

  /// \brief The raw op config, for fused norm+rope kernels that restate it.
  const ops::NormConfig& Config() const { return config_; }

 private:
  RMSNorm(Tensor weight, int64_t hidden_size, ops::NormConfig config)
      : weight_(std::move(weight)),
        hidden_size_(hidden_size),
        config_(config) {}

  Tensor weight_;  ///< [hidden]
  int64_t hidden_size_ = 0;
  ops::NormConfig config_;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_LAYERS_RMS_NORM_H_
