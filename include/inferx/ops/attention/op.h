#ifndef INFERX_ATTN_OP_H_
#define INFERX_ATTN_OP_H_

#include <cstdint>
#include <memory>

#include "inferx/core/status.h"
#include "inferx/ops/attention/kernel.h"
#include "inferx/ops/attention/spec.h"

namespace inferx::attn {

/// \brief Per-layer handle binding spec, kernel, cache view, and group
///        index: the single call site models use. Movable, not copyable
///        (owns its Kernel).
class AttentionOp {
 public:
  AttentionOp() = default;
  AttentionOp(AttentionOp&&) noexcept = default;
  AttentionOp& operator=(AttentionOp&&) noexcept = default;

  /// \brief One layer's attention for one step. At most one
  ///        function-pointer call (the separate KV writer) plus one
  ///        virtual call.
  [[nodiscard]] Status Forward(const StepContext& step, const Inputs& in,
                               TensorView out) const noexcept {
    const GroupMeta& meta = step.Group(group_);
    const RunArgs args{meta, cache_, step.Ctx(), in, out, step.Workspace(), step.desc};
    if (write_kv_ != nullptr && in.k.Defined()) {
      if (Status s = write_kv_(args); !s.ok()) return s;  // Layout-aware, no virtual.
    }
    return kernel_->Run(args);
  }

  [[nodiscard]] const AttentionSpec& GetSpec() const noexcept { return spec_; }
  [[nodiscard]] uint16_t GetGroup() const noexcept { return group_; }

 private:
  friend class AttentionRuntime;
  using KvWriteFn = Status (*)(const RunArgs&) noexcept;

  AttentionSpec spec_{};
  std::unique_ptr<Kernel> kernel_;
  KvWriteFn write_kv_ = nullptr;  // Non-null iff kernel is kSeparate.
  CacheView cache_{};
  std::uint16_t group_ = 0;
};

}  // namespace inferx::attn

#endif  // INFERX_ATTN_OP_H_
