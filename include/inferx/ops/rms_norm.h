#pragma once

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Parameters of one RMS normalization.
struct RMSNormConfig {
  /// \brief Added to each row's mean square before the reciprocal square
  ///        root; must be non-negative.
  float eps = 1e-6f;
  /// \brief Gemma-style variant: scale rows by (1 + weight) instead of
  ///        weight. Checkpoints that already fold the constant into the
  ///        weight leave this false.
  bool plus_one_weight = false;
  /// Round normalized activations to the input dtype before multiplying by
  /// weight, as in the Llama/Qwen reference implementations. The default
  /// retains the single-rounding mathematical operator (also used by Gemma).
  bool round_before_weight = false;
};

/// \brief RMS-normalizes each row of `x` into `out`.
///
/// Computes out[i, :] <- x[i, :] / sqrt(mean(x[i, :]^2) + eps) * scale, where
/// scale is `weight`, or `1 + weight` when `config.plus_one_weight` is set.
/// Rows are independent. `out` may alias `x` for in-place updates but must
/// not overlap `weight`. On accelerators work is enqueued on the context's
/// stream; tensors must not be read or written by anything else until that
/// stream completes.
///
/// Validation and backend selection live in this entry point; dtype support
/// is reported per backend and the kernel details stay inside them.
///
/// \param ctx     Execution context; all tensors must live on ctx.device().
/// \param x       [rows, dim] input activations.
/// \param weight  [dim] per-channel scale, same dtype as `x`.
/// \param out     [rows, dim] output tensor, same dtype and shape as `x`.
/// \param config  Normalization parameters.
/// \return        OK, or InvalidArgument/Unimplemented for bad inputs.
Status RmsNorm(ExecutionContext& ctx, const Tensor& x, const Tensor& weight, Tensor& out,
               const RMSNormConfig& config);

/// Add x into residual with activation-dtype rounding, then normalize residual
/// into out. Out must be separate from residual so both results are retained.
Status AddRmsNorm(ExecutionContext& ctx, const Tensor& x, Tensor& residual,
                  const Tensor& weight, Tensor& out, const RMSNormConfig& config);

}  // namespace inferx::ops
