#include "inferx/models/components/mlp.h"

#include "inferx/core/shape.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/linear.h"
#include "models/diagnostic_trace.h"

namespace inferx::components {

Status RunSwiGlu(const SwiGluConfig& config, const SwiGluWeights& weights,
                 const Tensor& normed, MlpWorkspace& ws, Tensor* packed_buffer,
                 ops::ExecutionContext& ctx, DiagnosticTrace* trace,
                 std::string_view prefix, Tensor& mixed_out) {
  const int64_t rows = normed.Dim(0);
  const bool tracing = trace != nullptr && trace->enabled();
  const auto write = [&](std::string_view stage, const Tensor& t) {
    if (tracing) trace->Write(std::string(prefix) + std::string(stage), t);
  };

  INFERX_ASSIGN_OR_RETURN(Tensor gate_flat, ws.gate->Slice(0, rows * config.intermediate_size));
  INFERX_ASSIGN_OR_RETURN(Tensor gate, gate_flat.Reshape(Shape({rows, config.intermediate_size})));
  // One fused GEMM over the packed gate/up weight, then a column de-interleave.
  const int64_t width = 2 * config.intermediate_size;
  INFERX_ASSIGN_OR_RETURN(auto flat, packed_buffer->Slice(0, rows * width));
  INFERX_ASSIGN_OR_RETURN(auto packed, flat.Reshape(Shape({rows, width})));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, normed, weights.packed_gate_up, packed));
  INFERX_RETURN_IF_ERROR(ops::PackedSiluAndMul(ctx, packed, gate));
  INFERX_RETURN_IF_ERROR(ops::Linear(ctx, gate, weights.down.weight, mixed_out));
  write("silu", gate);
  write("down_proj", mixed_out);
  return OkStatus();
}

}  // namespace inferx::components
