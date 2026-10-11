#pragma once

#include <optional>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/op_context.h"

namespace inferx::ops {

/// \brief Represents the kind of normalization operation to perform.
enum class NormKind {
  kRms,   ///< Root-mean-square norm; no mean subtraction, no bias.
  kLayer  ///< Layer norm; mean subtraction, optional bias.
};

/// \brief Represents the configuration of one normalization operation.
struct NormConfig {
  NormKind kind = NormKind::kRms;

  /// Epsilon added to the mean square (RMS) or variance (Layer) before the reciprocal square
  /// root. Must be positive.
  float eps = 1e-6f;

  /// RMS-only Gemma-style variant: scale rows by (1 + weight) instead of weight. Checkpoints
  /// that already fold the constant into the weight leave this false.
  bool plus_one = false;

  /// Round normalized activations to the input dtype before multiplying by
  /// the weight, as the Llama/Qwen reference implementations do (and, for
  /// the fused add form, normalize the dtype-rounded residual sum). The
  /// default retains the single-rounding fused operator (Gemma lineage).
  bool round_before_weight = false;

  /// LayerNorm-only: a bias tensor is supplied. Must agree with the presence of the bias
  /// argument itself.
  bool bias = false;
};

/// \brief RMS-normalizes each row of `x` into `out`.
///
/// Computes out[i, :] <- x[i, :] / sqrt(mean(x[i, :]^2) + eps) * scale, where
/// scale is `weight`, or `1 + weight` when `config.plus_one` is set. Rows are
/// independent. `out` may alias `x` for in-place updates but must not overlap
/// `weight`. On accelerators work is enqueued on the context's stream;
/// tensors must not be read or written by anything else until that stream
/// completes.
///
/// Validation and backend selection live in this entry point; dtype support
/// is reported per backend and the kernel details stay inside them.
///
/// \param ctx     Execution context; all tensors must live on ctx.GetDevice().
/// \param x       [rows, dim] input activations.
/// \param weight  [dim] per-channel scale, same dtype as `x`.
/// \param out     [rows, dim] output tensor, same dtype and shape as `x`.
/// \param config  Normalization parameters; kind must be kRms.
/// \return        OK, or InvalidArgument/Unimplemented for bad inputs.
Status RmsNorm(OpContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
               const NormConfig& config);

/// \brief Adds `x` and `residual`, stores the dtype-rounded sum in `residual`,
///        and RMS-normalizes the sum into `out`.
///
/// CUDA uses FlashInfer's fused operation: normalization uses the FP32 sum
/// before activation-dtype rounding. The CPU backend normalizes the rounded
/// residual, so low-precision outputs can differ between backends.
/// `out` must be distinct from `residual`; it may alias `x`. Otherwise `x`
/// is preserved (CUDA copies it into `out` before the fused operation).
Status AddRmsNorm(OpContext& ctx, const Tensor& x, Tensor& residual, const Tensor& weight,
                  Tensor& out, const NormConfig& config);

/// \brief Layer-normalizes each row of `x` into `out`.
///
/// Computes out[i, :] <- (x[i, :] - mean) / sqrt(var + eps) * weight + bias,
/// with the mean and variance taken over each row. `bias` participates only
/// when `config.bias` is set, and then must match `weight` in dtype and
/// length. `out` may alias `x` but must not overlap `weight` or `bias`.
Status LayerNorm(OpContext& ctx, const Tensor& x, const Tensor& weight,
                 const std::optional<Tensor>& bias, Tensor& out, const NormConfig& config);

}  // namespace inferx::ops
