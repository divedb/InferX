#ifndef INFERX_SERVER_UTF8_H_
#define INFERX_SERVER_UTF8_H_

#include <cstddef>
#include <string_view>

namespace inferx::server::detail {

/// Length of the prefix ending on a complete UTF-8 character boundary.
/// Keep the remaining bytes for the next detokenized suffix. This is a
/// boundary check, not UTF-8 validation; serialization replaces malformed
/// sequences separately.
inline std::size_t Utf8Boundary(std::string_view text) {
  if (text.empty()) return 0;
  std::size_t end = text.size();
  if (static_cast<unsigned char>(text[end - 1]) < 0x80) return end;
  std::size_t lead = end - 1;
  while (lead > 0 && (static_cast<unsigned char>(text[lead]) & 0xC0) == 0x80 &&
         end - lead < 4) {
    --lead;
  }
  const unsigned char first = static_cast<unsigned char>(text[lead]);
  std::size_t expected = 1;
  if ((first & 0xE0) == 0xC0) expected = 2;
  else if ((first & 0xF0) == 0xE0) expected = 3;
  else if ((first & 0xF8) == 0xF0) expected = 4;
  if (end - lead == expected) return end;
  return lead;
}

}  // namespace inferx::server::detail

#endif  // INFERX_SERVER_UTF8_H_
