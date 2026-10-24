// JSON codec unit tests: depth limit (64 accepted, 65 rejected), string
// length cap, UTF-8 validation, trailing garbage, node budget, and body
// size cap (docs §8.1, §13.1).
#include <string>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "inferx/server/json_codec.h"

namespace inferx::server {
namespace {

HttpLimits Limits() { return {}; }

std::string DeepNested(std::size_t depth) {
  return std::string(depth, '[') + "1" + std::string(depth, ']');
}

TEST(JsonCodecTest, ParsesValidObject) {
  auto parsed = ParseJson(R"({"a": 1, "b": [true, null, "x"]})", Limits(),
                          1024 * 1024);
  ASSERT_TRUE(parsed.ok()) << parsed.error().message;
  EXPECT_EQ(parsed.value()["a"], 1);
  EXPECT_EQ(parsed.value()["b"].size(), 3u);
}

TEST(JsonCodecTest, Depth64Accepted65Rejected) {
  HttpLimits limits;
  EXPECT_TRUE(ParseJson(DeepNested(64), limits, 1024 * 1024).ok());
  auto rejected = ParseJson(DeepNested(65), limits, 1024 * 1024);
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.error().http_status, 400);
  EXPECT_EQ(rejected.error().code, "invalid_json");
}

TEST(JsonCodecTest, StringLengthCapped) {
  HttpLimits limits;
  limits.max_json_string_bytes = 16;
  EXPECT_TRUE(ParseJson(R"("short")", limits, 1024).ok());
  EXPECT_FALSE(ParseJson("\"0123456789abcdefghij\"", limits, 1024).ok());
  EXPECT_FALSE(ParseJson(R"({"0123456789abcdefghij": 1})", limits, 1024).ok());
}

TEST(JsonCodecTest, InvalidUtf8Rejected) {
  const std::string invalid = "\"abc\x80\x81\"";
  auto parsed = ParseJson(invalid, Limits(), 1024);
  ASSERT_FALSE(parsed.ok());
  EXPECT_EQ(parsed.error().code, "invalid_json");
}

TEST(JsonCodecTest, TrailingGarbageRejected) {
  EXPECT_FALSE(ParseJson("{} trailing", Limits(), 1024).ok());
  EXPECT_FALSE(ParseJson("{}{}", Limits(), 1024).ok());
  EXPECT_FALSE(ParseJson("", Limits(), 1024).ok());
  EXPECT_TRUE(ParseJson("  42  ", Limits(), 1024).ok());
}

TEST(JsonCodecTest, BodySizeCappedBeforeParse) {
  HttpLimits limits;
  auto parsed = ParseJson("[1,2,3]", limits, 4);
  ASSERT_FALSE(parsed.ok());
  EXPECT_EQ(parsed.error().http_status, 413);
}

TEST(JsonCodecTest, Utf8Validator) {
  EXPECT_TRUE(IsValidUtf8("hello"));
  EXPECT_TRUE(IsValidUtf8("héllo 世界 👍"));
  EXPECT_FALSE(IsValidUtf8(std::string("bad\xFF")));
  // Overlong encoding (0xC0 0x80 = NUL).
  EXPECT_FALSE(IsValidUtf8(std::string("\xC0\x80", 2)));
  // Truncated multi-byte sequence.
  EXPECT_FALSE(IsValidUtf8(std::string("cut\xE2\x82", 4)));
  // Surrogate range code point (0xED 0xA0 0x80 = U+D800).
  EXPECT_FALSE(IsValidUtf8("\xED\xA0\x80"));
}

TEST(JsonCodecTest, DumpJsonRoundTrips) {
  auto parsed = ParseJson(R"({"k": [1, 2.5, "v"]})", Limits(), 1024);
  ASSERT_TRUE(parsed.ok());
  auto dumped = DumpJson(parsed.value());
  ASSERT_TRUE(dumped.ok());
  EXPECT_EQ(dumped.value(), R"({"k":[1,2.5,"v"]})");
}

}  // namespace
}  // namespace inferx::server
