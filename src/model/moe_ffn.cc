#include "inferx/model/moe_ffn.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include <cuda_runtime.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/kernels/gpt_oss.h"
#include "inferx/kernels/layers.h"
#include "inferx/kernels/moe.h"

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
    return TensorView::Create(buf.data(), dtype, shape, DeviceId::Cuda(0));
  }
};

Status Grow(Scratch* s, size_t bytes) {
  if (s->buf.size() >= bytes) return OkStatus();

  INFERX_ASSIGN_OR_RETURN(s->buf,
                          DeviceBuffer::Allocate(bytes, DeviceId::Cuda(0)));
  return OkStatus();
}

}  // namespace

struct MoeFfn::Impl {
  MoeFfn::Config config;

  int64_t capacity_tokens = 0;

  // Routing.
  Scratch router_logits;  // [tokens, E] bf16
  Scratch weights;        // [tokens, k] f32
  Scratch experts;        // [tokens, k] i32
  Scratch offsets;        // [E + 1] i32
  Scratch rows;           // [tokens * k] i32
  Scratch dest;           // [tokens * k] i32

  // Expert compute, all in grouped order.
  Scratch gathered;   // [tokens * k, hidden] bf16
  Scratch gate_up;    // [tokens * k, 2 * moe_inter] bf16
  Scratch activated;  // [tokens * k, moe_inter] bf16
  Scratch expert_out; // [tokens * k, hidden] bf16

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

  INFERX_RETURN_IF_ERROR(Grow(&router_logits,
                              bf16_size * tokens * config.num_experts));
  INFERX_RETURN_IF_ERROR(Grow(&weights, sizeof(float) * assignments));
  INFERX_RETURN_IF_ERROR(Grow(&experts, sizeof(int32_t) * assignments));
  INFERX_RETURN_IF_ERROR(
      Grow(&offsets, sizeof(int32_t) * (config.num_experts + 1)));
  INFERX_RETURN_IF_ERROR(Grow(&rows, sizeof(int32_t) * assignments));
  INFERX_RETURN_IF_ERROR(Grow(&dest, sizeof(int32_t) * assignments));

  INFERX_RETURN_IF_ERROR(Grow(&gathered, bf16_size * assignments * h));
  INFERX_RETURN_IF_ERROR(Grow(&gate_up, bf16_size * assignments * 2 * inter));
  INFERX_RETURN_IF_ERROR(Grow(&activated, bf16_size * assignments * inter));
  INFERX_RETURN_IF_ERROR(Grow(&expert_out, bf16_size * assignments * h));

  if (config.shared_intermediate > 0) {
    const int64_t si = config.shared_intermediate;
    INFERX_RETURN_IF_ERROR(Grow(&shared_gate_up, bf16_size * tokens * 2 * si));
    INFERX_RETURN_IF_ERROR(Grow(&shared_act, bf16_size * tokens * si));
    INFERX_RETURN_IF_ERROR(Grow(&shared_out, bf16_size * tokens * h));
    INFERX_RETURN_IF_ERROR(Grow(&shared_gate, bf16_size * tokens));
  }

  capacity_tokens = tokens;
  return OkStatus();
}

StatusOr<MoeFfn> MoeFfn::Create(const Config& config, int64_t max_tokens) {
  if (config.hidden <= 0 || config.moe_intermediate <= 0) {
    return InvalidArgumentError("MoeFfn: hidden and moe_intermediate must be "
                                "positive, got ", config.hidden, " and ",
                                config.moe_intermediate);
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

  INFERX_RETURN_IF_ERROR(impl->EnsureCapacity(max_tokens));

  return MoeFfn(std::move(impl));
}

MoeFfn::MoeFfn(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
MoeFfn::~MoeFfn() = default;
MoeFfn::MoeFfn(MoeFfn&&) noexcept = default;
MoeFfn& MoeFfn::operator=(MoeFfn&&) noexcept = default;

Status MoeFfn::Forward(const TensorView& x, const MoeWeights& weights,
                       const TensorView& out, kernels::CublasLtGemm* gemm,
                       cudaStream_t stream) {
  if (gemm == nullptr) return InvalidArgumentError("MoeFfn: gemm is null");

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
  INFERX_RETURN_IF_ERROR(
      gemm->LinearBF16(x, weights.router, logits_v, stream));

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
        kernels::AddBiasInPlace(logits_v, weights.router_bias, stream));
  }

  INFERX_RETURN_IF_ERROR(kernels::MoeRouteTopK(
      logits_v, weights_v, experts_v, c.norm_topk_prob, stream));

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

  INFERX_RETURN_IF_ERROR(kernels::MoeBuildDispatch(
      experts_v, c.num_experts, offsets_v, rows_v, dest_v, stream));

  INFERX_ASSIGN_OR_RETURN(
      const TensorView gathered_v,
      impl_->gathered.View(kBf16, Shape({assignments, c.hidden})));
  INFERX_RETURN_IF_ERROR(
      kernels::MoeGatherRows(x, rows_v, gathered_v, stream));

  // --- 3. One GEMM per expert ----------------------------------------------
  //
  // The offsets have to come back to the host: a GEMM's `m` is an argument to
  // a host-side cuBLASLt call, so the loop cannot be expressed without knowing
  // how many rows each expert got. That is the real cost of the per-expert
  // loop and the real reason a grouped GEMM exists -- it takes the counts as
  // device memory and never round-trips. Until then this is one D2H copy per
  // layer per step, which also makes the layer uncapturable as a graph.
  std::vector<int32_t> host_offsets(
      static_cast<size_t>(c.num_experts + 1));
  INFERX_CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(
      host_offsets.data(), offsets_v.Data(),
      sizeof(int32_t) * host_offsets.size(), cudaMemcpyDeviceToHost, stream));
  INFERX_CUDA_RETURN_IF_ERROR(cudaStreamSynchronize(stream));

  if (host_offsets.back() != assignments) {
    return InternalError("MoeFfn: dispatch placed ", host_offsets.back(),
                         " of ", assignments, " assignments");
  }

  INFERX_ASSIGN_OR_RETURN(
      const TensorView gate_up_all,
      impl_->gate_up.View(kBf16, Shape({assignments, 2 * inter})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView activated_all,
      impl_->activated.View(kBf16, Shape({assignments, inter})));
  INFERX_ASSIGN_OR_RETURN(
      const TensorView expert_out_all,
      impl_->expert_out.View(kBf16, Shape({assignments, c.hidden})));

  for (int64_t e = 0; e < c.num_experts; ++e) {
    const int64_t begin = host_offsets[static_cast<size_t>(e)];
    const int64_t end = host_offsets[static_cast<size_t>(e) + 1];
    const int64_t rows_here = end - begin;

    // An expert nobody routed to this step does no work at all. That is the
    // whole economic argument for MoE, and it is also why the loop cannot be
    // hoisted into a single GEMM with a block-diagonal weight.
    if (rows_here == 0) continue;

    INFERX_ASSIGN_OR_RETURN(const TensorView xe,
                            gathered_v.Slice(begin, end));
    INFERX_ASSIGN_OR_RETURN(const TensorView gu_e,
                            gate_up_all.Slice(begin, end));
    INFERX_ASSIGN_OR_RETURN(const TensorView act_e,
                            activated_all.Slice(begin, end));
    INFERX_ASSIGN_OR_RETURN(const TensorView ye,
                            expert_out_all.Slice(begin, end));

    // The stacked weights, viewed one expert at a time. Slice takes dimension
    // 0, which is the expert axis, and the result is [1, ...] -- reshaped down
    // to the 2-D operand the GEMM wants.
    INFERX_ASSIGN_OR_RETURN(const TensorView gu_w_3d,
                            weights.gate_up.Slice(e, e + 1));
    INFERX_ASSIGN_OR_RETURN(const TensorView gu_w,
                            gu_w_3d.Reshape(Shape({2 * inter, c.hidden})));

    INFERX_ASSIGN_OR_RETURN(const TensorView down_w_3d,
                            weights.down.Slice(e, e + 1));
    INFERX_ASSIGN_OR_RETURN(const TensorView down_w,
                            down_w_3d.Reshape(Shape({c.hidden, inter})));

    INFERX_RETURN_IF_ERROR(gemm->LinearBF16(xe, gu_w, gu_e, stream));

    if (weights.gate_up_bias.IsDefined()) {
      INFERX_ASSIGN_OR_RETURN(const TensorView b2d,
                              weights.gate_up_bias.Slice(e, e + 1));
      INFERX_ASSIGN_OR_RETURN(const TensorView b,
                              b2d.Reshape(Shape({2 * inter})));
      INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(gu_e, b, stream));
    }

    switch (c.activation) {
      case Activation::kSiluMul:
        INFERX_RETURN_IF_ERROR(kernels::SiluMulFused(gu_e, act_e, stream));
        break;
      case Activation::kGptOssClamped:
        INFERX_RETURN_IF_ERROR(kernels::GptOssSwiGlu(
            gu_e, act_e, c.swiglu_limit, c.swiglu_alpha, stream));
        break;
    }

    INFERX_RETURN_IF_ERROR(gemm->LinearBF16(act_e, down_w, ye, stream));

    if (weights.down_bias.IsDefined()) {
      INFERX_ASSIGN_OR_RETURN(const TensorView b2d,
                              weights.down_bias.Slice(e, e + 1));
      INFERX_ASSIGN_OR_RETURN(const TensorView b,
                              b2d.Reshape(Shape({c.hidden})));
      INFERX_RETURN_IF_ERROR(kernels::AddBiasInPlace(ye, b, stream));
    }
  }

  // --- 4. Combine, then the shared expert -----------------------------------

  INFERX_RETURN_IF_ERROR(kernels::MoeCombineRows(expert_out_all, dest_v,
                                                 weights_v, out, stream));

  if (c.shared_intermediate > 0) {
    if (!weights.shared_gate_up.IsDefined() ||
        !weights.shared_down.IsDefined() || !weights.shared_gate.IsDefined()) {
      return InvalidArgumentError("MoeFfn: shared_intermediate is ",
                                  c.shared_intermediate,
                                  " but the shared expert's weights are unset");
    }

    const int64_t si = c.shared_intermediate;

    INFERX_ASSIGN_OR_RETURN(
        const TensorView sgu,
        impl_->shared_gate_up.View(kBf16, Shape({tokens, 2 * si})));
    INFERX_ASSIGN_OR_RETURN(
        const TensorView sact,
        impl_->shared_act.View(kBf16, Shape({tokens, si})));
    INFERX_ASSIGN_OR_RETURN(
        const TensorView sout,
        impl_->shared_out.View(kBf16, Shape({tokens, c.hidden})));
    INFERX_ASSIGN_OR_RETURN(
        const TensorView sgate,
        impl_->shared_gate.View(kBf16, Shape({tokens, 1})));

    INFERX_RETURN_IF_ERROR(
        gemm->LinearBF16(x, weights.shared_gate_up, sgu, stream));
    INFERX_RETURN_IF_ERROR(kernels::SiluMulFused(sgu, sact, stream));
    INFERX_RETURN_IF_ERROR(
        gemm->LinearBF16(sact, weights.shared_down, sout, stream));
    INFERX_RETURN_IF_ERROR(
        gemm->LinearBF16(x, weights.shared_gate, sgate, stream));

    INFERX_RETURN_IF_ERROR(
        kernels::MoeAddSharedExpert(sout, sgate, out, stream));
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

  INFERX_CUDA_RETURN_IF_ERROR(
      cudaMemcpy(offsets.data(), impl_->offsets.buf.data(),
                 sizeof(int32_t) * offsets.size(), cudaMemcpyDeviceToHost));

  std::vector<int32_t> counts(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    counts[static_cast<size_t>(e)] = offsets[static_cast<size_t>(e) + 1] -
                                     offsets[static_cast<size_t>(e)];
  }

  return counts;
}

}  // namespace inferx::model
