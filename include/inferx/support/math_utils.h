#pragma once

#include <concepts>
#include <optional>

namespace inferx {

/// \brief Overflow-checked addition for integral types.
///
/// Wraps `__builtin_add_overflow` (GCC/Clang) so a call site can express an
/// overflow-safe sum as "the result, or nothing" instead of a hand-rewritten
/// comparison that needs a comment to prove it cannot wrap.
///
/// \tparam  T The integral type.
/// \param a The first operand.
/// \param b The second operand.
/// \return  The sum of `a` and `b`, or `std::nullopt` if overflow occurs.
template <typename T>
  requires std::integral<T> && (!std::same_as<T, bool>)
std::optional<T> CheckedAdd(T a, T b) noexcept {
  T result;

  if (__builtin_add_overflow(a, b, &result)) {
    return std::nullopt;
  }

  return result;
}

}  // namespace inferx
