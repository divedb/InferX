#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

/// Unicode support for the tokenizer, and only for the tokenizer.
///
/// The scope here is deliberately narrow: exactly the operations HF's tokenizer
/// performs on Qwen2 input, which is a UTF-8 codec, the `\p{L}` and `\p{N}`
/// character classes the pre-tokenizer regex uses, and NFC normalization.  This
/// is not a general Unicode library and should not grow into one -- anything
/// beyond what the tokenizer needs belongs in a real dependency.
namespace inferx::tokenizer::unicode {

/// The codepoint substituted for malformed input.
///
/// Decoding never fails.  Serving takes bytes off a socket, and a request with
/// one bad byte in it should come back with a slightly mangled completion, not
/// an error -- which is also what HF does, since Rust's `String` has already
/// replaced the bad bytes by the time the tokenizer sees them.
inline constexpr uint32_t kReplacementChar = 0xFFFD;

/// \brief Decodes UTF-8 to codepoints, substituting U+FFFD for bad sequences.
///
/// Rejects overlong encodings, surrogates, and anything above U+10FFFF, so the
/// output is always well-formed Unicode regardless of the input.
std::vector<uint32_t> Utf8Decode(std::string_view text);

/// \brief Appends the UTF-8 encoding of `cp` to `out`.
///
/// Surrogates and out-of-range values are encoded as U+FFFD rather than
/// producing invalid UTF-8.
void Utf8Encode(uint32_t cp, std::string* out);

/// \brief Encodes a codepoint sequence to UTF-8.
std::string Utf8Encode(const std::vector<uint32_t>& codepoints);

/// \brief Length in bytes of the UTF-8 sequence starting with `lead`.
///
/// Returns 0 if `lead` is not a valid leading byte.  Used by the streaming
/// decoder to decide whether a trailing partial sequence must be held back.
int Utf8SequenceLength(unsigned char lead);

/// \brief Whether `text` ends in the middle of a UTF-8 sequence.
///
/// The streaming path needs this because a token boundary is not a character
/// boundary: one BPE token can carry the first two bytes of a three-byte
/// character, and emitting them alone would put a replacement char into the
/// stream that a non-streaming decode of the same tokens would not contain.
bool EndsWithPartialUtf8(std::string_view text);

/// \brief Whether `cp` is in general category L* (`\p{L}`).
bool IsLetter(uint32_t cp);

/// \brief Whether `cp` is in general category N* (`\p{N}`).
bool IsNumber(uint32_t cp);

/// \brief Whether `cp` is whitespace as Rust's regex crate defines `\s`.
///
/// This is `\p{White_Space}`, which is wider than C's `isspace` -- it includes
/// NEL, the Unicode spaces, and the line and paragraph separators.  Getting
/// this wrong would mis-split text containing non-breaking spaces, which is
/// common enough in web-scraped input to matter.
bool IsWhitespace(uint32_t cp);

/// \brief Normalizes `text` to NFC.
///
/// Canonical decomposition, canonical ordering by combining class, then
/// canonical composition, per UAX #15.  Hangul is handled arithmetically.
///
/// Fast path: text already in NFC is returned unchanged without allocating a
/// codepoint buffer, which is nearly all real input.
std::string NormalizeNfc(std::string_view text);

}  // namespace inferx::tokenizer::unicode
