// Derived from LLVM's llvm/ADT/ScopeExit.h.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <type_traits>
#include <utility>

namespace inferx {
namespace detail {

template <typename Callable>
class ScopeExit {
 public:
  template <typename Function>
  explicit ScopeExit(Function&& function)
      : exit_function_(std::forward<Function>(function)) {}

  ScopeExit(ScopeExit&& other)
      : exit_function_(std::move(other.exit_function_)),
        engaged_(other.engaged_) {
    other.Release();
  }

  ScopeExit(const ScopeExit&) = delete;
  ScopeExit& operator=(ScopeExit&&) = delete;
  ScopeExit& operator=(const ScopeExit&) = delete;

  ~ScopeExit() {
    if (engaged_) exit_function_();
  }

  void Release() { engaged_ = false; }

 private:
  Callable exit_function_;
  bool engaged_ = true;
};

}  // namespace detail

/// Executes `function` when the returned guard leaves scope unless released.
template <typename Callable>
[[nodiscard]] detail::ScopeExit<std::decay_t<Callable>> MakeScopeExit(
    Callable&& function) {
  return detail::ScopeExit<std::decay_t<Callable>>(
      std::forward<Callable>(function));
}

}  // namespace inferx
