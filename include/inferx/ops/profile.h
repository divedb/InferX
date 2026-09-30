#pragma once
#include <cstdio>
#include <cstdlib>

#include "inferx/ops/op_context.h"
namespace inferx::ops {
// Diagnostic only: serializes the named GPU op, never enabled in benchmarks.
// CUDA events are a fallback where WSL Nsight lacks GPU activity records.
template <class F>
Status ProfileCall(OpContext& ctx, const char* name, F&& f) {
  static const bool enabled = std::getenv("INFERX_PROFILE_OPS") != nullptr;
  if (!enabled || ctx.Device().kind != DeviceKind::kCuda) return f();
  INFERX_ASSIGN_OR_RETURN(auto start, ctx.Runtime().CreateEvent(true));
  INFERX_ASSIGN_OR_RETURN(auto end, ctx.Runtime().CreateEvent(true));
  auto status = ctx.Runtime().RecordEvent(start, ctx.GetStream());
  if (status.ok()) status = f();
  if (status.ok()) status = ctx.Runtime().RecordEvent(end, ctx.GetStream());
  if (status.ok()) status = ctx.Runtime().SynchronizeEvent(end);
  if (status.ok()) {
    auto ms = ctx.Runtime().ElapsedMs(start, end);
    if (ms.ok())
      std::fprintf(stderr, "PROFILE,%s,%.6f\n", name, *ms);
    else
      status = ms.status();
  }
  (void)ctx.Runtime().DestroyEvent(start);
  (void)ctx.Runtime().DestroyEvent(end);
  return status;
}
}  // namespace inferx::ops
