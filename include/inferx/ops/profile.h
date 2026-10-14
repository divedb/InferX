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
  if (!enabled || ctx.GetDevice().Type() != DeviceType::kCuda) return f();
  auto& streams = ctx.Runtime().Streams();
  INFERX_ASSIGN_OR_RETURN(auto start, streams.CreateEvent(true));
  INFERX_ASSIGN_OR_RETURN(auto end, streams.CreateEvent(true));
  auto status = streams.RecordEvent(start.Get(), ctx.GetStream());
  if (status.ok()) status = f();
  if (status.ok()) status = streams.RecordEvent(end.Get(), ctx.GetStream());
  if (status.ok()) status = streams.SynchronizeEvent(end.Get());
  if (status.ok()) {
    auto ms = streams.ElapsedMs(start.Get(), end.Get());
    if (ms.ok())
      std::fprintf(stderr, "PROFILE,%s,%.6f\n", name, *ms);
    else
      status = ms.status();
  }
  return status;
}
}  // namespace inferx::ops
