#include "inferx/model/moe_ffn.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "inferx/core/device_buffer.h"
#include "inferx/core/device_runtime.h"
#include "inferx/model/parallel/linear.h"
#include "inferx/ops/gpt_oss.h"
#include "inferx/ops/layers.h"
#include "inferx/ops/moe.h"
#include "inferx/ops/mxfp4_gemm.h"

namespace inferx::model {
namespace {

constexpr DataType kBf16 = DataType::kBFloat16;

// A device allocation plus the views taken over it. Same shape as the one in
// qwen2.cc, and kept local for the same reason it is local there: a scratch
// buffer's only contract is "big enough and alive", and sharing an abstraction
// for that between two files buys nothing.
struct Scratch {
  DeviceBuffer buf;

  StatusOr<TensorView> View(DataType dtype, const Shape& shape) const {
    return TensorView::Create(buf.data(), dtype, shape, buf.device());
  }
};

Status Grow(Scratch* s, size_t bytes, DeviceId device) {
  if (s->buf.size() >= bytes) return OkStatus();

  INFERX_ASSIGN_OR_RETURN(s->buf, DeviceBuffer::Allocate(bytes, device));
  return OkStatus();
}

}  // namespace

struct MoeFfn::Impl {
  MoeFfn::Config config;
  DeviceId device;
  DeviceRuntime* runtime = nullptr;

  int64_t capacity_tokens = 0;

  // Routing.
  Scratch router_logits;  // [tokens, E] bf16
  Scratch weights;        // [tokens, k] f32
  Scratch experts;        // [tokens, k] i32
  Scratch offsets;        // [E + 1] i32
  Scratch rows;           // [tokens * k] i32
  Scratch dest;           // [tokens * k] i32

  // Expert compute, all in grouped order.
  Scratch gathered;    // [tokens * k, hidden] bf16
  Scratch gate_up;     // [tokens * k, 2 * moe_inter] bf16
  Scratch activated;   // [tokens * k, moe_inter] bf16
  Scratch expert_out;  // [tokens * k, hidden] bf16

  // The shared expert, which is dense over every token rather than grouped.
  Scratch shared_gate_up;  // [tokens, 2 * shared_inter] bf16
  Scratch shared_act;      // [tokens, shared_inter] bf16
  Scratch shared_out;      // [tokens, hidden] bf16
  Scratch shared_gate;     // [tokens, 1] bf16

  Status EnsureCapacity(int64_t tokens);
};

Status MoeFfn::Impl::EnsureCapacity(int64_t tokens) {
  if (tokens <= capacity_tokens) return OkStatus();

  const int64_t k = config.top_k;
  const int64_t h = config.hidden;
  const int64_t inter = config.moe_intermediate;
  const int64_t assignments = tokens * k;
  const size_t bf16_size = DataTypeByteSize(kBf16, 1);

  INFERX_RETURN_IF_ERROR(
      Grow(&router_logits, bf16_size * tokens * config.num_experts, device));
  INFERX_RETURN_IF_ERROR(Grow(&weights, sizeof(float) * assignments, device));
  INFERX_RETURN_IF_ERROR(Grow(&experts, sizeof(int32_t) * assignments, device));
  INFERX_RETURN_IF_ERROR(
      Grow(&offsets, sizeof(int32_t) * (config.num_experts + 1), device));
  INFERX_RETURN_IF_ERROR(Grow(&rows, sizeof(int32_t) * assignments, device));
  INFERX_RETURN_IF_ERROR(Grow(&dest, sizeof(int32_t) * assignments, device));

  INFERX_RETURN_IF_ERROR(Grow(&gathered, bf16_size * assignments * h, device));
  INFERX_RETURN_IF_ERROR(
      Grow(&gate_up, bf16_size * assignments * 2 * inter, device));
  INFERX_RETURN_IF_ERROR(
      Grow(&activated, bf16_size * assignments * inter, device));
  INFERX_RETURN_IF_ERROR(
      Grow(&expert_out, bf16_size * assignments * h, device));

  if (config.shared_intermediate > 0) {
    const int64_t si = config.shared_intermediate;
    INFERX_RETURN_IF_ERROR(
        Grow(&shared_gate_up, bf16_size * tokens * 2 * si, device));
    INFERX_RETURN_IF_ERROR(Grow(&shared_act, bf16_size * tokens * si, device));
    INFERX_RETURN_IF_ERROR(Grow(&shared_out, bf16_size * tokens * h, device));
    INFERX_RETURN_IF_ERROR(Grow(&shared_gate, bf16_size * tokens, device));
  }

  capacity_tokens = tokens;
  return OkStatus();
}

StatusOr<MoeFfn> MoeFfn::Create(const Config& config, int64_t max_tokens,
                                DeviceId device) {
  if (config.hidden <= 0 || config.moe_intermediate <= 0) {
    return InvalidArgumentError(
        "MoeFfn: hidden and moe_intermediate must be "
        "positive, got ",
        config.hidden, " and ", config.moe_intermediate);
  }

  if (config.num_experts <= 0 || config.top_k <= 0 ||
      config.top_k > config.num_experts) {
    return InvalidArgumentError("MoeFfn: cannot route to ", config.top_k,
                                " of ", config.num_experts, " experts");
  }

  if (max_tokens <= 0) {
    return InvalidArgumentError("MoeFfn: max_tokens must be positive, got ",
                                max_tokens);
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->device = device;
  INFERX_ASSIGN_OR_RETURN(impl->runtime, RuntimeFor(device));

  INFERX_RETURN_IF_ERROR(impl->EnsureCapacity(max_tokens));

  return MoeFfn(std::move(impl));
}

MoeFfn::MoeFfn(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MoeFfn::~MoeFfn() = default;
MoeFfn::MoeFfn(MoeFfn&&) noexcept = default;
MoeFfn& MoeFfn::operator=(MoeFfn&&) noexcept = default;

Status MoeFfn::Forward(const TensorView& x, const MoeWeights& weights,
                       const TensorView& out, ops::CublasLtGemm* gemm,
                       Stream stream) {
  return ForwardImpl(x, weights, out, gemm, nullptr, stream);
}

Status MoeFfn::ForwardParallel(const TensorView& x, const MoeWeights& weights,
                               const TensorView& out, ops::CublasLtGemm* gemm,
                               comm::Communicator& communicator,
                               Stream stream) {
  return ForwardImpl(x, weights, out, gemm, &communicator, stream);
}

Status MoeFfn::ForwardImpl(const TensorView& x, const MoeWeights& weights,
                           const TensorView& out, ops::CublasLtGemm* gemm,
                           comm::Communicator* communicator, Stream stream) {
  if (gemm == nullptr) return InvalidArgumentError("MoeFfn: gemm is null");

  const bool defer_down_bias = communicator != nullptr &&
                               communicator->size() > 1 &&
                               weights.down_bias.IsDefined();
  const TensorView local_down_bias =
      defer_down_bias ? TensorView{} : weights.down_bias;

  if (!x.IsDefined() || x.Rank() != 2 || x.GetDataType() != kBf16) {
    return InvalidArgumentError("MoeFfn: x must be a 2-D bf16 tensor");
  }

  const Config& c = impl_->config;
  const int64_t tokens = x.Dim(0);

  if (x.Dim(1) != c.hidden) {
    return InvalidArgumentError("MoeFfn: x is ", x.Dim(1), " wide, expected ",
                                c.hidden);
  }

  if (out.Rank() != 2 || out.Dim(0) != tokens || out.Dim(1) != c.hidden) {
    return InvalidArgumentError("MoeFfn: out must be [", tokens, ", ", c.hidden,
                                "], got ", out.GetShape().ToString());
  }

  if (out.Data() == x.Data()) {
    return InvalidArgumentError("MoeFfn: out must not alias x");
  }

  if (tokens == 0) return OkStatus();

  INFERX_RETURN_IF_ERROR(impl_->EnsureCapacity(tokens));

  const int64_t k = c.top_k;
  const int64_t assignments = tokens * k;
  const int64_t inter = c.moe_intermediate;

  // --- 1. Route -------------------------------------------------------------

  INFERX_ASSIGN_OR_RETURN(
      const TensorView logits_v,
      impl_->router_logits.View(kBf16, Shape({tokens, c.num_experts})));
  INFERX_RETURN_IF_ERROR(gemm->LinearBF16(x, weights.router, logits_v, stream));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView weights_v,
      impl_->weights.View(DataType::kFloat, Shape({tokens, k})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView experts_v,
      impl_->experts.View(DataType::kInt32, Shape({tokens, k})));

  // The router's bias, when the architecture has one. Added before the top-k,
  // because it shifts which experts win and not merely by how much.
  if (weights.router_bias.IsDefined()) {
    INFERX_RETURN_IF_ERROR(
        ops::AddBiasInPlace(logits_v, weights.router_bias, stream));
  }

  INFERX_RETURN_IF_ERROR(ops::MoeRouteTopK(logits_v, weights_v, experts_v,
                                           c.norm_topk_prob,
                                           c.routed_scaling_factor, stream));

  // --- 2. Group by expert ---------------------------------------------------

  INFERX_ASSIGN_OR_RETURN(
      const TensorView offsets_v,
      impl_->offsets.View(DataType::kInt32, Shape({c.num_experts + 1})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView rows_v,
      impl_->rows.View(DataType::kInt32, Shape({assignments})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView dest_v,
      impl_->dest.View(DataType::kInt32, Shape({assignments})));

  INFERX_RETURN_IF_ERROR(ops::MoeBuildDispatch(
      experts_v, c.num_experts, offsets_v, rows_v, dest_v, stream));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView gathered_v,
      impl_->gathered.View(kBf16, Shape({assignments, c.hidden})));
  INFERX_RETURN_IF_ERROR(ops::MoeGatherRows(x, rows_v, gathered_v, stream));

  // --- 3. Expert projections ------------------------------------------------
  INFERX_ASSIGN_OR_RETURN(
      const TensorView gate_up_all,
      impl_->gate_up.View(kBf16, Shape({assignments, 2 * inter})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView activated_all,
      impl_->activated.View(kBf16, Shape({assignments, inter})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView expert_out_all,
      impl_->expert_out.View(kBf16, Shape({assignments, c.hidden})));

  const bool grouped_mxfp4 = weights.gate_up_blocks.IsDefined();
  if (grouped_mxfp4) {
    // Both ragged projections consume device-resident offsets. Bias is fused
    // into each projection, leaving only one dense activation launch between
    // them and no host involvement in expert dispatch.
    INFERX_RETURN_IF_ERROR(ops::Mxfp4GroupedGemm(
        gathered_v, offsets_v, weights.gate_up_blocks, weights.gate_up_scales,
        weights.gate_up_bias, gate_up_all,
        /*deinterleave=*/true, stream));
    switch (c.activation) {
      case Activation::kSiluMul:
        INFERX_RETURN_IF_ERROR(
            ops::SiluMulFused(gate_up_all, activated_all, stream));
        break;
      case Activation::kGptOssClamped:
        INFERX_RETURN_IF_ERROR(ops::GptOssSwiGlu(gate_up_all, activated_all,
                                                 c.swiglu_limit, c.swiglu_alpha,
                                                 stream));
        break;
    }
    INFERX_RETURN_IF_ERROR(ops::Mxfp4GroupedGemm(
        activated_all, offsets_v, weights.down_blocks, weights.down_scales,
        local_down_bias, expert_out_all, /*deinterleave=*/false, stream));
  } else {
    // The bf16 grouped path: both ragged projections consume the
    // device-resident offsets, so nothing round-trips to the host and the
    // launch count does not scale with E. This replaced a per-expert
    // cuBLASLt loop whose offsets readback was one D2H sync per layer per
    // step -- and the reason the bf16 FFN could not be graph-captured.
    // Bias is fused into each projection, as on the MXFP4 path.
    INFERX_RETURN_IF_ERROR(
        ops::MoeGroupedGemmBf16(gathered_v, offsets_v, weights.gate_up,
                                weights.gate_up_bias, gate_up_all, stream));
    switch (c.activation) {
      case Activation::kSiluMul:
        INFERX_RETURN_IF_ERROR(
            ops::SiluMulFused(gate_up_all, activated_all, stream));
        break;
      case Activation::kGptOssClamped:
        INFERX_RETURN_IF_ERROR(ops::GptOssSwiGlu(gate_up_all, activated_all,
                                                 c.swiglu_limit, c.swiglu_alpha,
                                                 stream));
        break;
    }
    INFERX_RETURN_IF_ERROR(
        ops::MoeGroupedGemmBf16(activated_all, offsets_v, weights.down,
                                local_down_bias, expert_out_all, stream));
  }

  // --- 4. Combine, then the shared expert -----------------------------------

  INFERX_RETURN_IF_ERROR(
      ops::MoeCombineRows(expert_out_all, dest_v, weights_v, out, stream));

  if (c.shared_intermediate > 0) {
    if (!weights.shared_gate_up.IsDefined() ||
        !weights.shared_down.IsDefined()) {
      return InvalidArgumentError("MoeFfn: shared_intermediate is ",
                                  c.shared_intermediate,
                                  " but the shared expert's weights are unset");
    }

    // The gate weight must match the convention exactly. A gated config with
    // no gate cannot compute; an ungated config with a gate means the loader
    // read a Qwen2-MoE checkpoint as DeepSeek (or vice versa), and silently
    // ignoring the tensor would hide it.
    if (c.shared_gated != weights.shared_gate.IsDefined()) {
      return InvalidArgumentError(
          "MoeFfn: shared expert is ", c.shared_gated ? "gated" : "ungated",
          " but shared_gate is ",
          weights.shared_gate.IsDefined() ? "set" : "unset");
    }

    const int64_t si = c.shared_intermediate;

    INFERX_ASSIGN_OR_RETURN(
        const TensorView sgu,
        impl_->shared_gate_up.View(kBf16, Shape({tokens, 2 * si})));
    INFERX_ASSIGN_OR_RETURN(const TensorView sact,
                            impl_->shared_act.View(kBf16, Shape({tokens, si})));
    INFERX_ASSIGN_OR_RETURN(
        const TensorView sout,
        impl_->shared_out.View(kBf16, Shape({tokens, c.hidden})));

    INFERX_RETURN_IF_ERROR(
        gemm->LinearBF16(x, weights.shared_gate_up, sgu, stream));
    INFERX_RETURN_IF_ERROR(ops::SiluMulFused(sgu, sact, stream));
    INFERX_RETURN_IF_ERROR(
        gemm->LinearBF16(sact, weights.shared_down, sout, stream));

    if (c.shared_gated) {
      INFERX_ASSIGN_OR_RETURN(
          const TensorView sgate,
          impl_->shared_gate.View(kBf16, Shape({tokens, 1})));
      INFERX_RETURN_IF_ERROR(
          gemm->LinearBF16(x, weights.shared_gate, sgate, stream));
      INFERX_RETURN_IF_ERROR(ops::MoeAddSharedExpert(sout, sgate, out, stream));
    } else {
      // DeepSeek adds the shared experts unconditionally; there is no gate.
      INFERX_RETURN_IF_ERROR(ops::AddInPlace(out, sout, stream));
    }
  }

  if (communicator != nullptr) {
    INFERX_RETURN_IF_ERROR(
        parallel::RowParallelLinear::ReduceOutput(*communicator, out, stream));
  }
  if (defer_down_bias) {
    INFERX_RETURN_IF_ERROR(ops::MoeAddExpertBias(
        experts_v, weights_v, weights.down_bias, out, stream));
  }

  return OkStatus();
}

StatusOr<std::vector<int32_t>> MoeFfn::LastExpertCounts() const {
  const int64_t num_experts = impl_->config.num_experts;

  std::vector<int32_t> offsets(static_cast<size_t>(num_experts + 1));

  if (impl_->offsets.buf.size() <
      sizeof(int32_t) * static_cast<size_t>(num_experts + 1)) {
    return FailedPreconditionError("MoeFfn: no Forward has run yet");
  }

  INFERX_RETURN_IF_ERROR(impl_->runtime->Copy(
      offsets.data(), impl_->offsets.buf.data(),
      sizeof(int32_t) * offsets.size(), CopyKind::kDeviceToHost));

  std::vector<int32_t> counts(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    counts[static_cast<size_t>(e)] =
        offsets[static_cast<size_t>(e) + 1] - offsets[static_cast<size_t>(e)];
  }

  return counts;
}

}  // namespace inferx::model
