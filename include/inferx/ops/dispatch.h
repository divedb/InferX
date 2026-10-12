#ifndef INFERX_OPS_DISPATCH_H_
#define INFERX_OPS_DISPATCH_H_

#include <initializer_list>
#include <utility>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/ops/op_context.h"

namespace inferx::ops {

/// \brief Implementations of one op, grouped by device kind. Entries for the
///        same kind are tried in registration order.
template <typename Fn>
class KernelTable {
 public:
  struct Entry {
    DeviceKind kind;
    Fn fn;
  };

  KernelTable(std::initializer_list<Entry> entries) : entries_(entries) {}

  const std::vector<Entry>& GetEntries() const { return entries_; }

 private:
  std::vector<Entry> entries_;
};

/// \brief Runs the first implementation for the context's device that does
///        not decline.
///
/// An implementation declines by returning Unimplemented (unsupported dtype,
/// shape, or architecture). It must do so *before* any side effect, because
/// the next candidate then runs on the same buffers. Any other status,
/// including errors, is final and returned as is. If every candidate declines,
/// the last one's status is returned; if none exists for the device, an
/// Unimplemented error naming the op and device is returned.
///
/// Implementations are called as fn(ctx, args...); `args` are passed as
/// lvalues so they can be reused across candidates.
template <typename Fn, typename... Args>
Status Dispatch(OpContext& ctx, const char* op, const KernelTable<Fn>& table, Args&&... args) {
  const DeviceKind kind = ctx.GetDevice().Type();

  for (const auto& entry : table.GetEntries()) {
    if (entry.kind != kind) continue;

    return entry.fn(ctx, args...);
  }

  return UnimplementedError(op, " has no implementation for device ", ctx.GetDevice().ToString());
}

}  // namespace inferx::ops

#endif  // INFERX_OPS_DISPATCH_H_