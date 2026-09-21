#pragma once

#include <cstddef>
#include <optional>

namespace inferx {

/// \brief Adds two sizes, reporting overflow instead of wrapping.
///
/// \param a The first addend.
/// \param b The second addend.
/// \return  The sum, or `std::nullopt` if the addition overflows `size_t`.
constexpr std::optional<size_t> CheckedAdd(size_t a, size_t b) {
  const size_t sum = a + b;
  if (sum < a) {
    return std::nullopt;
  }
  return sum;
}

}  // namespace inferx
