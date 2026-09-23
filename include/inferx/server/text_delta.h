#ifndef INFERX_SERVER_TEXT_DELTA_H_
#define INFERX_SERVER_TEXT_DELTA_H_

#include <cstddef>
#include <string_view>

#include "inferx/server/utf8.h"

namespace inferx::server {

/// \brief Incremental text delta over full re-decodes of a growing token
/// list.
///
/// The gateway re-decodes the whole generated list every step and streams
/// the new suffix. A full re-decode is not prefix-stable: a byte-level
/// decoder renders a trailing partial UTF-8 character as U+FFFD
/// (`String::from_utf8_lossy`), and the character that later completes it
/// can be shorter than the replacement it replaces. This tracker therefore
/// emits only settled prefixes: everything up to trailing replacement
/// characters, which the next decode may rewrite. Replacement characters
/// that are settled (more text follows them) are permanent — the invalid
/// bytes they stand for can never become valid again — and are emitted
/// normally. Bytes still held back at the end of a request are returned by
/// Flush().
class TextDelta {
 public:
  /// Returns the suffix of `text` that is safe to emit now, advancing the
  /// internal offset. Empty when nothing new has settled.
  std::string_view NextDelta(std::string_view text) {
    // Decode cannot shrink the settled prefix, but a defensive re-sync
    // beats the alternative if some decoder ever violates that: the
    // 2026-09-22 crash was exactly substr() past the end of a decode that
    // had shrunk below the emitted offset.
    if (emitted_ > text.size()) emitted_ = text.size();
    std::size_t end = detail::Utf8Boundary(text);
    while (end - emitted_ >= 3 && text.substr(end - 3, 3) == kReplacement) {
      end -= 3;
    }
    if (end == emitted_) return {};
    const std::size_t begin = emitted_;
    emitted_ = end;
    return text.substr(begin, end - begin);
  }

  /// Returns everything not yet emitted, held-back replacement characters
  /// included; for the final event of a request.
  std::string_view Flush(std::string_view text) {
    if (emitted_ > text.size()) emitted_ = text.size();
    const std::size_t begin = emitted_;
    emitted_ = text.size();
    return text.substr(begin);
  }

  /// Bytes already emitted.
  std::size_t emitted() const { return emitted_; }

 private:
  static constexpr std::string_view kReplacement = "\xEF\xBF\xBD";

  std::size_t emitted_ = 0;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_TEXT_DELTA_H_
