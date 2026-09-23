#pragma once
#include <cstdio>
#include <cstdlib>

#include "inferx/ops/execution_context.h"
namespace inferx::ops {
// Diagnostic only: serializes the named GPU op, never enabled in benchmarks.
// CUDA events are a fallback where WSL Nsight lacks GPU activity records.
template <class F>
Status ProfileCall(ExecutionContext& ctx, const char* name, F&& f) {
  static const bool enabled = std::getenv("INFERX_PROFILE_OPS") != nullptr;
  if (!enabled || ctx.device().kind != DeviceKind::kCuda) return f();
  INFERX_ASSIGN_OR_RETURN(auto start, ctx.runtime().CreateEvent(true));
  INFERX_ASSIGN_OR_RETURN(auto end, ctx.runtime().CreateEvent(true));
  auto status = ctx.runtime().RecordEvent(start, ctx.stream());
  if (status.ok()) status = f();
  if (status.ok()) status = ctx.runtime().RecordEvent(end, ctx.stream());
  if (status.ok()) status = ctx.runtime().SynchronizeEvent(end);
  if (status.ok()) {
    auto ms = ctx.runtime().ElapsedMs(start, end);
    if (ms.ok())
      std::fprintf(stderr, "PROFILE,%s,%.6f\n", name, *ms);
    else
      status = ms.status();
  }
  (void)ctx.runtime().DestroyEvent(start);
  (void)ctx.runtime().DestroyEvent(end);
  return status;
}
}  // namespace inferx::ops
