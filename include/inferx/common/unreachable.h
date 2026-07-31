#pragma once

#include <cassert>

namespace inferx {

/// \brief Marks a program point the design assumes can never be reached.
///
/// Use it to close out an exhausted `switch` over an enum, or a branch the
/// surrounding logic guarantees cannot be taken, so the compiler knows there is
/// no fall-through and the assumption is greppable at every site. It is the
/// C++20 stand-in for C++23 `std::unreachable`, which our GCC 13 floor does
/// not provide.
///
/// Reaching it is undefined behavior. In debug builds the `assert` fires first
/// with a diagnostic; in release builds it lowers to `__builtin_unreachable`,
/// which is the point -- the optimizer may assume the call site is dead. Use a
/// real error path (`absl::StatusOr`, a validated reject) anywhere the
/// assumption could fail under adversarial or evolving input.
[[noreturn]] inline void Unreachable() {
  assert(false && "Unreachable() was reached");

    __builtin_unreachable();
}

}  // namespace inferx
