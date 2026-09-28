#include "inferx/ops/moe.h"

#include <vector>

#include "inferx/ops/cuda/moe.h"

namespace inferx::ops {
namespace {

Status CheckHost(const Tensor& t, DataType dtype, const DeviceId& device) {
  if (t.GetDataType() != dtype || t.Device() != device) {
    return InvalidArgumentError("MoE routing tensors must be ", DataTypeName(dtype), " on ",
                                device.ToString());
  }
  return OkStatus();
}

}  // namespace

Status RouteTokens(ExecutionContext& ctx, const Tensor& hidden, const Tensor& router_weight,
                   const Tensor* router_bias, const Tensor* correction_bias,
                   const RoutingConfig& config, Tensor& topk_indices, Tensor& topk_weights) {
  const DeviceId device = ctx.device();
  if (hidden.Rank() != 2 || router_weight.Rank() != 2 ||
      hidden.GetDataType() != DataType::kBFloat16 ||
      router_weight.GetDataType() != DataType::kBFloat16 ||
      router_weight.Dim(1) != hidden.Dim(1) || router_weight.Dim(0) <= 0) {
    return InvalidArgumentError("router expects [rows, hidden] activations and [experts, hidden] weights");
  }
  for (const Tensor* t : {&topk_indices, &topk_weights}) {
    if (t->Rank() != 1 || t->Numel() != hidden.Dim(0) * config.topk) {
      return InvalidArgumentError("router outputs must hold rows x topk entries");
    }
  }
  INFERX_RETURN_IF_ERROR(CheckHost(topk_indices, DataType::kInt32, device));
  INFERX_RETURN_IF_ERROR(CheckHost(topk_weights, DataType::kFloat32, device));
  if (router_bias != nullptr &&
      (router_bias->Rank() != 1 || router_bias->Dim(0) != router_weight.Dim(0) ||
       router_bias->GetDataType() != DataType::kBFloat16)) {
    return InvalidArgumentError("router bias must be [experts] bfloat16");
  }
  if (correction_bias != nullptr &&
      (correction_bias->Rank() != 1 || correction_bias->Dim(0) != router_weight.Dim(0) ||
       correction_bias->GetDataType() != DataType::kFloat32)) {
    return InvalidArgumentError("correction bias must be [experts] float32");
  }
  if (config.topk <= 0 || config.topk > router_weight.Dim(0) || config.group_count <= 0 ||
      router_weight.Dim(0) % config.group_count != 0 || config.group_topk < 0 ||
      config.group_topk > config.group_count) {
    return InvalidArgumentError("routing configuration disagrees with expert count");
  }
  if (config.group_count > 1 &&
      config.group_topk * (router_weight.Dim(0) / config.group_count) < config.topk) {
    return InvalidArgumentError("grouped routing cannot select enough experts");
  }
  for (const Tensor* t : std::initializer_list<const Tensor*>{
           &hidden, &router_weight, &topk_indices, &topk_weights}) {
    if (t->Device() != device) {
      return InvalidArgumentError("router tensors must live on the context device");
    }
  }
  if (!device.IsCuda()) return UnimplementedError("routing requires CUDA");
  if (hidden.IsEmpty()) return OkStatus();
  return cuda::RouteTokens(ctx, hidden, router_weight, router_bias, correction_bias, config,
                           topk_indices, topk_weights);
}

StatusOr<std::vector<int64_t>> BuildExpertDispatch(
    ExecutionContext& ctx, const Tensor& topk_indices, const Tensor& topk_weights,
    int64_t num_experts, Tensor& counts, Tensor& offsets, Tensor& cursor, Tensor& token_rows,
    Tensor& weights_by_slot) {
  const int64_t total = topk_indices.Numel();
  if (counts.Numel() != num_experts || offsets.Numel() != num_experts + 1 ||
      cursor.Numel() != num_experts || token_rows.Numel() != total ||
      weights_by_slot.Numel() != total) {
    return InvalidArgumentError("dispatch buffers disagree with the routed batch");
  }
  // Histogram, sync the counts to the host (expert GEMMs need row counts),
  // upload the prefix offsets, and only then scatter slots: the scatter
  // writes through those offsets, so ordering is load-bearing.
  INFERX_RETURN_IF_ERROR(cuda::ExpertHistogram(ctx, topk_indices, num_experts, counts));
  std::vector<int32_t> host(num_experts);
  INFERX_RETURN_IF_ERROR(ctx.runtime().CopyAsync(
      host.data(), counts.Data(), num_experts * sizeof(int32_t), CopyKind::kDeviceToHost,
      ctx.stream()));
  INFERX_RETURN_IF_ERROR(ctx.runtime().SynchronizeStream(ctx.stream()));
  std::vector<int64_t> host_offsets(num_experts + 1);
  host_offsets[0] = 0;
  for (int64_t e = 0; e < num_experts; ++e) {
    host_offsets[e + 1] = host_offsets[e] + host[e];
  }
  if (host_offsets.back() != total) {
    return InternalError("expert dispatch lost routed slots");
  }
  std::vector<int32_t> device_offsets(num_experts + 1);
  for (int64_t e = 0; e <= num_experts; ++e) {
    device_offsets[e] = static_cast<int32_t>(host_offsets[e]);
  }
  INFERX_RETURN_IF_ERROR(ctx.runtime().CopyAsync(offsets.Data(), device_offsets.data(),
                                                 device_offsets.size() * sizeof(int32_t),
                                                 CopyKind::kHostToDevice, ctx.stream()));
  INFERX_RETURN_IF_ERROR(cuda::ScatterSlots(ctx, topk_indices, topk_weights, offsets, cursor,
                                            token_rows, weights_by_slot));
  return host_offsets;
}

Status GatherRoutedTokens(ExecutionContext& ctx, const Tensor& hidden, const Tensor& token_rows,
                          Tensor& out) {
  if (hidden.Rank() != 2 || token_rows.Rank() != 1 || out.Rank() != 2 ||
      out.Dim(0) < token_rows.Numel() || out.Dim(1) != hidden.Dim(1) ||
      hidden.GetDataType() != DataType::kBFloat16 || out.GetDataType() != DataType::kBFloat16) {
    return InvalidArgumentError("gather expects [slots, hidden] output for [slots] rows");
  }
  if (token_rows.Numel() == 0) return OkStatus();
  return cuda::GatherRoutedTokens(ctx, hidden, token_rows, out);
}

Status ScatterRoutedOutputs(ExecutionContext& ctx, const Tensor& expert_rows,
                            const Tensor& token_rows, const Tensor& weights_by_slot,
                            int64_t begin, int64_t count, Tensor& out) {
  if (count <= 0) return InvalidArgumentError("scatter needs a positive expert slice");
  if (expert_rows.Rank() != 2 || out.Rank() != 2 || expert_rows.Dim(1) != out.Dim(1) ||
      begin < 0 || begin + count > expert_rows.Dim(0)) {
    return InvalidArgumentError("scatter shapes disagree with the expert slice");
  }
  return cuda::ScatterRoutedOutputs(ctx, expert_rows, token_rows, weights_by_slot, begin, count,
                                    out);
}

}  // namespace inferx::ops
