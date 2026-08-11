#pragma once

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor_view.h"

/// The kernels a mixture-of-experts FFN needs that a dense one does not.
///
/// A dense FFN is one GEMM per projection over every token. An MoE FFN routes
/// each token to `k` of `E` experts, so the work is ragged: expert `e` sees
/// however many tokens chose it, which is a runtime quantity. Everything here
/// exists to turn that ragged problem back into dense GEMMs — group the tokens
/// that share an expert, run one GEMM per group, and put the results back where
/// they came from.
///
/// The pipeline, for `T` tokens and `A = T·k` (token, slot) assignments:
///
///     RouteTopK      logits [T,E]        -> weights [T,k], experts [T,k]
///     BuildDispatch  experts [T,k]       -> offsets [E+1], rows [A], dest [A]
///     GatherRows     x [T,H], rows [A]   -> xg [A,H]
///        (caller runs one GEMM per expert over xg[offsets[e] : offsets[e+1]])
///     CombineRows    yg [A,H], weights   -> out [T,H]
///
/// **Determinism is a requirement here, not a preference** (§6.3 and R8): the
/// engine's contract is that identical requests produce identical tokens, and a
/// float sum whose order depends on thread scheduling breaks it. So no kernel
/// below uses an atomic to place or to accumulate a value. `BuildDispatch`
/// produces a *stable* grouping — within an expert, tokens stay in token order
/// — and `CombineRows` sums each token's `k` contributions in slot order from a
/// single thread block. Both are the reason the obvious atomic-cursor
/// implementations are not what is here.
namespace inferx::ops {

/// \brief Softmax over experts, then the top `k` of them.
///
/// The Mixtral/Qwen2-MoE routing rule: softmax the router's logits across all
/// experts, take the `k` largest, and optionally renormalize those `k` to sum
/// to 1. Renormalizing is `norm_topk_prob` in the config — it changes the
/// output, so it is a parameter rather than a convention.
///
/// Ties are broken towards the lower expert index, which makes the routing a
/// pure function of the logits. That matters because two experts with bitwise
/// equal scores is not a hypothetical at bf16 precision.
///
/// \param logits      `[tokens, num_experts]` bf16, straight from the gate
/// GEMM. \param out_weights `[tokens, k]` fp32, the gate value for each chosen
/// expert,
///                    descending by score. fp32 because it multiplies an
///                    expert's whole output and a bf16 gate would quantize the
///                    mixture itself.
/// \param out_experts `[tokens, k]` int32, the chosen expert indices.
/// \param renormalize Divide the `k` weights by their sum.
/// \param scale       Multiplies every written weight — DeepSeek's
///                    `routed_scaling_factor`, applied where HF applies it: to
///                    the gate weights, so the routed mixture scales and the
///                    shared expert (added later) does not. 1 everywhere else.
Status MoeRouteTopK(const TensorView& logits, const TensorView& out_weights,
                    const TensorView& out_experts, bool renormalize,
                    float scale = 1.0f, Stream stream = {});

/// \brief Groups the `tokens × k` assignments by expert, stably.
///
/// \param experts     `[tokens, k]` int32 from `MoeRouteTopK`.
/// \param num_experts `E`. Every value in `experts` must be in `[0, E)`.
/// \param out_offsets `[E + 1]` int32. Expert `e` owns rows
///                    `[offsets[e], offsets[e+1])` of the grouped array, so
///                    `offsets[E] == tokens · k` and a count is a subtraction.
/// \param out_rows    `[tokens · k]` int32. For each grouped row, the token it
///                    came from — the gather index.
/// \param out_dest    `[tokens · k]` int32. The inverse: for assignment
///                    `t·k + slot`, the grouped row it landed in. This is what
///                    lets `MoeCombineRows` read its `k` contributions without
///                    searching for them, and what keeps the combine
///                    atomic-free.
Status MoeBuildDispatch(const TensorView& experts, int64_t num_experts,
                        const TensorView& out_offsets,
                        const TensorView& out_rows, const TensorView& out_dest,
                        Stream stream = {});

/// \brief `out[i, :] = x[rows[i], :]`.
///
/// The permutation that makes each expert's tokens contiguous, so the expert's
/// GEMM is an ordinary dense GEMM over a slice.
///
/// \param x    `[tokens, width]` bf16.
/// \param rows `[n]` int32, indices into `x`'s first dimension.
/// \param out  `[n, width]` bf16.
Status MoeGatherRows(const TensorView& x, const TensorView& rows,
                     const TensorView& out, Stream stream = {});

/// \brief Undoes the permutation and mixes: `out[t,:] = Σ_slot w[t,slot] ·
///        y[dest[t·k + slot], :]`.
///
/// \param y       `[tokens · k, width]` bf16, the experts' outputs in grouped
///                order.
/// \param dest    `[tokens · k]` int32 from `MoeBuildDispatch`.
/// \param weights `[tokens, k]` fp32 from `MoeRouteTopK`.
/// \param out     `[tokens, width]` bf16. Overwritten, not accumulated into.
///                A caller wanting a residual or a shared expert adds it
///                afterwards, because folding that in here would make the
///                kernel's contract depend on what the caller left in `out`.
Status MoeCombineRows(const TensorView& y, const TensorView& dest,
                      const TensorView& weights, const TensorView& out,
                      Stream stream = {});

/// \brief Adds the routed mixture of per-expert output biases to `out`.
///
/// Kept separate from grouped GEMM so tensor parallelism can apply each
/// replicated bias exactly once after reducing rank-local projection partials.
Status MoeAddExpertBias(const TensorView& experts, const TensorView& weights,
                        const TensorView& bias, const TensorView& out,
                        Stream stream = {});

/// \brief One ragged GEMM over stacked bf16 expert weights, offsets on the
///        device.
///
/// The bf16 twin of `Mxfp4GroupedGemm`: rows of `x` are grouped by expert
/// (from `MoeBuildDispatch`), `w` is the stacked `[E, n, k]` expert tensor,
/// and each row multiplies its own expert's slice — with the expert ranges
/// read from device memory, so nothing round-trips to the host and the launch
/// count does not scale with E. This is what retired the per-expert cuBLASLt
/// loop and its per-layer D2H sync, which was also the bf16 MoE path's
/// graph-capture blocker. A CUTLASS grouped GEMM is the named upgrade for
/// large prefill shapes; the shape of this call would be its operand layout.
///
/// \param x       `[assignments, k]` bf16, grouped by expert.
/// \param offsets `[E + 1]` int32 on the device, from `MoeBuildDispatch`.
/// \param w       `[E, n, k]` bf16, stacked expert weights.
/// \param bias    `[E, n]` bf16 or undefined.
/// \param y       `[assignments, n]` bf16, written.
Status MoeGroupedGemmBf16(const TensorView& x, const TensorView& offsets,
                          const TensorView& w, const TensorView& bias,
                          const TensorView& y, Stream stream = {});

/// \brief `out = sigmoid(gate_logit) · shared`, elementwise over each row.
///
/// Qwen2-MoE's shared expert: one FFN every token passes through, scaled by its
/// own scalar gate. Separate from `MoeCombineRows` because the shared expert
/// does not participate in the top-k mixture at all — it is added to it.
///
/// \param shared      `[tokens, width]` bf16, the shared expert's output.
/// \param gate_logits `[tokens, 1]` bf16, pre-sigmoid.
/// \param out         `[tokens, width]` bf16, accumulated into (`out += ...`).
Status MoeAddSharedExpert(const TensorView& shared,
                          const TensorView& gate_logits, const TensorView& out,
                          Stream stream = {});

}  // namespace inferx::ops
