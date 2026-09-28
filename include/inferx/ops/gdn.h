#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Splits a per-group [q | k | v | z] projection into block layout.
///
/// The checkpoint orders rows per key-head group; the kernels want
/// [q (rows x key_heads*key_dim) | k | v | z] block-contiguous. `conv_in`
/// receives [q | k | v] in one buffer (the causal convolution's input) and
/// `z` its own buffer.
Status SplitGdnProjection(ExecutionContext& ctx, const Tensor& packed, Tensor& conv_in,
                          Tensor& z, int64_t key_heads, int64_t key_dim, int64_t value_heads,
                          int64_t value_dim);

/// \brief Depthwise causal convolution with silu, updating the conv state.
///
/// `x` is [tokens, channels] over the flat batch; `weight` is
/// [channels, kernel]; `state` is [slots, channels, kernel - 1] inputs from
/// previous steps. `batch_indices` maps each token row to its sequence and
/// `qo_indptr` bounds the sequential walk. Writes the convolved+activated
/// output over `x` in place and leaves the last kernel-1 inputs in state.
Status GdnCausalConv(ExecutionContext& ctx, Tensor& x, const Tensor& weight,
                     const Tensor& state, const Tensor& batch_indices,
                     const Tensor& qo_indptr, int64_t kernel);

/// \brief Computes the per-head gates from the [b | a] projection.
///
/// `ba` is [tokens, 2 * value_heads] in per-group [b | a] order (the
/// checkpoint layout); beta = sigmoid(b) and
/// g = -exp(a_log) * softplus(a + dt_bias) run in fp32. `a_log` and
/// `dt_bias` are [value_heads] float32.
Status GdnGates(ExecutionContext& ctx, const Tensor& ba, const Tensor& a_log,
                const Tensor& dt_bias, Tensor& beta, Tensor& g);

/// \brief Sequential gated-delta update over the flat token batch.
///
/// Per (sequence, value head, dv tile): normalize q/k (eps 1e-6), scale q by
/// 1/sqrt(dk), then for each token S <- exp(g) S; S <- S + beta k (v - S k)^T;
/// y <- S q. State is [slots, value_heads, key_dim, value_dim] fp32 indexed
/// by `slot_indices` per sequence. `conv` is the convolved [q | k | v] buffer
/// ([tokens, 2 * query_width + value_width]) with `query_width` the total
/// per-k-head key columns; `y` is [tokens, value_heads * value_dim]. 
Status GdnRecurrent(ExecutionContext& ctx, const Tensor& conv, int64_t query_width,
                    const Tensor& beta, const Tensor& g, Tensor& state,
                    const Tensor& slot_indices, const Tensor& qo_indptr,
                    const Tensor& batch_indices, Tensor& y);

/// \brief Per-head gated RMS norm: out = rmsnorm(y) * weight * silu(z).
///
/// `y` and `z` are [tokens, value_heads * value_dim]; `weight` is
/// [value_dim]; heads are independent.
Status RmsNormGated(ExecutionContext& ctx, const Tensor& y, const Tensor& z,
                    const Tensor& weight, float eps, Tensor& out);

}  // namespace inferx::ops
