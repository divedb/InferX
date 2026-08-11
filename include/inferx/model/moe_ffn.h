#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/core/stream.h"
#include "inferx/core/tensor_view.h"
#include "inferx/ops/gemm.h"

namespace inferx::model {

/// \brief The weights of one mixture-of-experts FFN layer.
///
/// Expert weights are **stacked**, not held as `num_experts` separate tensors:
/// `gate_up` is `[E, 2·moe_inter, hidden]` and `down` is `[E, hidden,
/// moe_inter]`, one contiguous allocation each. Two reasons, and the second is
/// the load-bearing one. A checkpoint stores them per expert, so stacking is a
/// load-time copy that costs nothing at run time; and an expert's GEMM then
/// wants a *view* at a fixed offset rather than a lookup, which is what makes
/// the per-expert loop below a loop over offsets and what a grouped GEMM would
/// want as its operand layout when one arrives (§9, T-CUTLASS).
struct MoeWeights {
  /// `[hidden, num_experts]`ᵀ as the GEMM sees it: `[num_experts, hidden]`.
  TensorView router;

  /// `[num_experts, 2 · moe_intermediate, hidden]`, gate and up concatenated
  /// along the output dimension exactly as the dense path fuses them.
  TensorView gate_up;

  /// `[num_experts, hidden, moe_intermediate]`.
  TensorView down;

  // The shared expert, which every token passes through. Undefined tensors
  // mean the architecture has none, which is the Mixtral case.

  /// `[2 · shared_intermediate, hidden]`.
  TensorView shared_gate_up;
  /// `[hidden, shared_intermediate]`.
  TensorView shared_down;
  /// `[1, hidden]` — the shared expert's own scalar gate, pre-sigmoid.
  /// Qwen2-MoE only; DeepSeek's ungated shared experts leave it undefined
  /// (\see Config::shared_gated).
  TensorView shared_gate;

  // Biases. Undefined means the architecture has none, which is Qwen2-MoE and
  // Mixtral; gpt-oss biases all three.

  /// `[num_experts]`, added to the router's logits before the top-k.
  TensorView router_bias;
  /// `[num_experts, 2 · moe_intermediate]`.
  TensorView gate_up_bias;
  /// `[num_experts, hidden]`.
  TensorView down_bias;

  // MXFP4 expert weights, for the fused grouped-GEMM path. When defined,
  // MoeFfn::Forward consumes the device dispatch offsets directly, without a
  // host readback or one launch per expert. gpt-oss is the only user today;
  // the bf16 path stays default and untouched.
  //
  // Shape per expert: gate_up_blocks is [2*inter, hidden/2] u8, gate_up_scales
  // is [2*inter, hidden/32] u8, down similarly. The expert axis is handled by
  // the caller's Slice, same as the bf16 path.
  TensorView gate_up_blocks, gate_up_scales;
  TensorView down_blocks, down_scales;
};

/// \brief A mixture-of-experts FFN: route, group, one GEMM per expert, combine.
///
/// The replacement for the dense `gate_up → SiLU·mul → down` sequence when a
/// checkpoint declares experts. What it does per call:
///
///   1. `router` GEMM → `[tokens, E]` logits, softmax + top-k → `k` experts
///      per token with their gate weights.
///   2. Group the `tokens·k` assignments by expert, so each expert's rows are
///      contiguous.
///   3. **One ragged grouped GEMM per projection**, offsets on the device —
///      `MoeGroupedGemmBf16` for stacked bf16 experts, `Mxfp4GroupedGemm` for
///      gpt-oss's packed 4-bit ones. Neither path round-trips the expert
///      counts to the host, so the FFN is graph-capturable and the launch
///      count does not scale with E. A CUTLASS grouped GEMM remains the named
///      upgrade for large prefill shapes (§9, T-CUTLASS) and would slot in
///      behind the same call shape.
///   4. Combine each token's `k` outputs by gate weight, and add the shared
///      expert if the architecture has one.
///
/// TP is not implemented here and the shape is deliberate about where it goes:
/// §7.3 chooses **TP over experts' weights**, where every rank holds a slice of
/// every expert's `gate_up`/`down` and the combine is followed by one
/// all-reduce — so a TP rank runs exactly this code with narrower
/// `moe_intermediate` and one comm call appended. Expert parallelism, where a
/// rank owns whole experts and tokens are all-to-all'd, would instead replace
/// steps 2–4; that is why the grouping is a separate call from the GEMM loop
/// rather than fused into it.
class MoeFfn {
 public:
  /// \brief Which gated activation the experts use.
  enum class Activation {
    /// `silu(gate) · up`, what Mixtral and Qwen2-MoE use.
    kSiluMul,
    /// gpt-oss's `(up+1) · gate · σ(α·gate)` with an asymmetric clamp.
    /// \see kernels/gpt_oss.h for why every part of that differs.
    kGptOssClamped,
  };

  struct Config {
    int64_t hidden = 0;
    int64_t num_experts = 0;
    int64_t top_k = 0;
    int64_t moe_intermediate = 0;
    int64_t shared_intermediate = 0;  // 0 when there is no shared expert
    bool norm_topk_prob = true;

    /// Whether the shared expert is gated by its own sigmoid
    /// (`out += σ(gate)·shared`, Qwen2-MoE's convention) or added plainly
    /// (`out += shared`, DeepSeek's). Gated requires `shared_gate`; ungated
    /// must leave it undefined — a checkpoint carrying a gate the config says
    /// to ignore is a mis-read, not a preference.
    bool shared_gated = true;

    /// DeepSeek's `routed_scaling_factor`, multiplying the routed mixture's
    /// gate weights and never the shared expert. 1.0 — every non-DeepSeek
    /// architecture — is the identity.
    float routed_scaling_factor = 1.0f;

    Activation activation = Activation::kSiluMul;
    /// Only read for `kGptOssClamped`.
    float swiglu_limit = 7.0f;
    float swiglu_alpha = 1.702f;
  };

  /// \brief Builds the layer and sizes its scratch for `max_tokens`.
  ///
  /// Scratch is allocated once rather than per call, for the reason §6.1 gives:
  /// freeing accelerator memory in the steady state may synchronize the
  /// device. Scratch grows on
  /// demand and never shrinks, so the first prefill sets the high-water mark.
  static StatusOr<MoeFfn> Create(const Config& config, int64_t max_tokens,
                                 DeviceId device);

  ~MoeFfn();
  MoeFfn(const MoeFfn&) = delete;
  MoeFfn& operator=(const MoeFfn&) = delete;
  MoeFfn(MoeFfn&&) noexcept;
  MoeFfn& operator=(MoeFfn&&) noexcept;

  /// \brief `out = MoE(x)`, for `x` `[tokens, hidden]` bf16.
  ///
  /// `out` may not alias `x`: the combine writes rows in token order while the
  /// experts are still reading gathered copies, and the aliasing case is not
  /// worth the check it would need per row.
  ///
  /// \param gemm The GEMM the expert projections run through. Passed rather
  ///             than owned because plan caches are per device and per process,
  ///             and every layer sharing one is the point of that cache.
  Status Forward(const TensorView& x, const MoeWeights& weights,
                 const TensorView& out, ops::CublasLtGemm* gemm,
                 Stream stream = {});

  /// \brief The per-expert row counts of the last `Forward`, for tests and
  ///        for the load-imbalance metric a scheduler would want.
  ///
  /// Copies from the device, so it synchronizes. Not for the serving path.
  StatusOr<std::vector<int32_t>> LastExpertCounts() const;

 private:
  struct Impl;
  explicit MoeFfn(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::model
