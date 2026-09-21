#pragma once

#include <cstddef>
#include <cstdint>

namespace inferx {

/// \brief Rounds `value` up to the next multiple of `alignment`.
///
/// `alignment` must be a non-zero power of two. Works for both `size_t` byte
/// counts and `uintptr_t` addresses; the return type matches `value` so call
/// sites do not have to cast back.
///
/// \param value     The value to round up.
/// \param alignment The alignment, a non-zero power of two.
/// \return          The smallest multiple of `alignment` that is >= `value`.
template <typename T>
constexpr T AlignUp(T value, size_t alignment) {
  const uintptr_t mask = static_cast<uintptr_t>(alignment) - 1u;

  return static_cast<T>((static_cast<uintptr_t>(value) + mask) & ~mask);
}

}  // namespace inferx
