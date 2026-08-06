#pragma once

#include <cstdint>

namespace inferx {

/// \brief Rounds `n` up to the next multiple of `alignment`.
///
/// \param n         The value to round up.
/// \param alignment The alignment, which must be a non-zero power of two.
/// \return          The smallest multiple of `alignment` that is >= `n`.
constexpr size_t AlignUp(size_t n, size_t alignment) {
  return (n + alignment - 1) & ~(alignment - 1);
}

}  // namespace inferx
