// Incremental-delta protocol over full re-decodes of a growing token list.
// A full re-decode is not prefix-stable: HF tokenizers render a trailing
// partial UTF-8 character as U+FFFD (`String::from_utf8_lossy` in
// ByteLevel::decode_chain), and the character that later completes it can
// be shorter than the replacement it replaces — so the decoded text can
// shrink between steps. The delta tracker must emit chunks that concatenate
// to the final decode and must never throw, whatever the decode sequence.
#include "inferx/server/text_delta.h"

#include <string>
#include <vector>

#include "gtest/gtest.h"

namespace inferx::server {
namespace {

// The UTF-8 replacement character, EF BF BD: what a byte-level decoder
// substitutes for a partial multi-byte character.
constexpr const char* kReplacement = "\xEF\xBF\xBD";

// Feeds `decodes` through the tracker and returns the concatenated output
// plus the final flush, so every test asserts the same end-to-end property.
std::string StreamAll(TextDelta& delta, const std::vector<std::string>& decodes,
                      bool flush = true) {
  std::string out;
  for (const std::string& text : decodes) {
    out += std::string(delta.NextDelta(text));
  }
  if (flush) out += std::string(delta.Flush(decodes.back()));
  return out;
}

TEST(TextDeltaTest, AsciiGrowthEmitsEachSuffix) {
  TextDelta delta;
  EXPECT_EQ(delta.NextDelta("Hel"), "Hel");
  EXPECT_EQ(delta.NextDelta("Hello"), "lo");
  EXPECT_EQ(delta.NextDelta("Hello world"), " world");
  EXPECT_EQ(delta.Flush("Hello world"), "");
  EXPECT_EQ(delta.emitted(), 11u);
}

// The 2026-09-22 server abort: a 3-byte character split across generated
// tokens. The first decode ends in a replacement character (three bytes);
// the completed character is also three bytes, so the text does not shrink,
// but emitting the replacement corrupted the stream and the offset then
// ran past the shorter completion of a 2-byte character (see below).
TEST(TextDeltaTest, SplitThreeByteCharNeverEmitsTheReplacement) {
  TextDelta delta;
  // "你" = E4 BD A0; after two of its three tokens the decode ends in U+FFFD.
  EXPECT_EQ(delta.NextDelta("hi \xEF\xBF\xBD"), "hi ");
  EXPECT_EQ(delta.NextDelta("hi \xE4\xBD\xA0"), "\xE4\xBD\xA0");
  EXPECT_EQ(delta.Flush("hi \xE4\xBD\xA0"), "");
}

// The exact field signature: `substr: __pos (which is 1107) > __size
// (which is 1106)`. A 2-byte character ("é" = C3 A9) split across two
// tokens: the partial prefix decodes to U+FFFD (three bytes), the completed
// character is two bytes — the decode shrinks below the emitted offset.
TEST(TextDeltaTest, SplitTwoByteCharShrinkDoesNotThrow) {
  TextDelta delta;
  EXPECT_EQ(delta.NextDelta("w\xEF\xBF\xBD"), "w");
  // Must not throw std::out_of_range; emits the completed character.
  EXPECT_EQ(delta.NextDelta("w\xC3\xA9"), "\xC3\xA9");
  EXPECT_EQ(delta.Flush("w\xC3\xA9"), "");
}

TEST(TextDeltaTest, SplitFourByteEmojiHeldBackUntilComplete) {
  TextDelta delta;
  // 😀 = F0 9F 98 80, one byte per generated token.
  EXPECT_EQ(delta.NextDelta("a\xEF\xBF\xBD"), "a");
  EXPECT_EQ(delta.NextDelta("a\xEF\xBF\xBD"), "");
  EXPECT_EQ(delta.NextDelta("a\xF0\x9F\x98\x80"), "\xF0\x9F\x98\x80");
  EXPECT_EQ(delta.Flush("a\xF0\x9F\x98\x80"), "");
}

TEST(TextDeltaTest, MultipleTrailingReplacementsAreHeldBack) {
  TextDelta delta;
  EXPECT_EQ(delta.NextDelta("ab\xEF\xBF\xBD\xEF\xBF\xBD"), "ab");
  EXPECT_EQ(delta.NextDelta("ab\xC3\xA9\xE4\xBD\xA0"), "\xC3\xA9\xE4\xBD\xA0");
  EXPECT_EQ(delta.Flush("ab\xC3\xA9\xE4\xBD\xA0"), "");
}

// Replacement characters that are settled (followed by more text) are
// permanent: the invalid bytes they stand for can never become valid by
// appending tokens, so they must be emitted, not held back forever.
TEST(TextDeltaTest, SettledReplacementInMiddleIsEmitted) {
  TextDelta delta;
  const std::string text = std::string("x\xEF\xBF\xBD") + "y";
  EXPECT_EQ(delta.NextDelta(text), text);
  EXPECT_EQ(delta.Flush(text), "");
}

// Held-back bytes surface at the finish, so a finished request's deltas
// concatenate to exactly its final full decode.
TEST(TextDeltaTest, FlushEmitsHeldBackRemainder) {
  TextDelta delta;
  EXPECT_EQ(delta.NextDelta("tail\xEF\xBF\xBD"), "tail");
  EXPECT_EQ(delta.Flush("tail\xEF\xBF\xBD"), "\xEF\xBF\xBD");
  EXPECT_EQ(delta.emitted(), 7u);
}

TEST(TextDeltaTest, ConcatenatedDeltasEqualFinalDecode) {
  // Byte-growth of "ok é 你 😀" through partial-character prefixes.
  const std::vector<std::string> decodes = {
      "ok", "ok ", "ok \xEF\xBF\xBD", "ok \xC3\xA9", "ok \xC3\xA9 ",
      "ok \xC3\xA9 \xEF\xBF\xBD", "ok \xC3\xA9 \xE4\xBD\xA0",
      "ok \xC3\xA9 \xE4\xBD\xA0 \xEF\xBF\xBD",
      "ok \xC3\xA9 \xE4\xBD\xA0 \xF0\x9F\x98\x80",
  };
  TextDelta delta;
  EXPECT_EQ(StreamAll(delta, decodes), decodes.back());
}

}  // namespace
}  // namespace inferx::server
