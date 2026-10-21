// Tokenizer facade tests (docs/tokenizer_design.md §8.1): the compatibility
// surface round-trips against the tiny fixture, option overloads refuse
// silently-degraded behavior with Unimplemented, and batching matches
// per-item encoding.
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/status.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx {
namespace {

constexpr const char* kTokenizerPath = "tests/testdata/tiny_tokenizer/tokenizer.json";

TEST(TokenizerTest, LoadsAndReportsVocab) {
  auto tokenizer = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  EXPECT_GT((*tokenizer)->vocab_size(), 0);

  const TokenizerInfo info = (*tokenizer)->Info();
  EXPECT_EQ(info.vocab_size, (*tokenizer)->vocab_size());
}

TEST(TokenizerTest, EncodeThenDecodeIsTotalAndDeterministic) {
  auto tokenizer = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  // The tiny fixture is a toy vocabulary (unknown text becomes <unk>), so
  // these assertions check totality and determinism rather than text
  // round-trip fidelity, which the fixture cannot express.
  for (const std::string& text : std::vector<std::string>{
           "", "abc", "hello world", "héllo 世界", std::string(4096, 'a')}) {
    const auto ids = (*tokenizer)->Encode(text);
    ASSERT_TRUE(ids.ok()) << ids.status();
    if (!text.empty()) {
      EXPECT_FALSE(ids->empty()) << "non-empty text must encode";
    }
    const auto again = (*tokenizer)->Encode(text);
    ASSERT_TRUE(again.ok()) << again.status();
    EXPECT_EQ(*ids, *again) << "encode must be deterministic";

    const auto decoded = (*tokenizer)->Decode(*ids);
    ASSERT_TRUE(decoded.ok()) << decoded.status();
  }
}

TEST(TokenizerTest, UnimplementedOptionsAreExplicit) {
  auto tokenizer = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  EncodeOptions with_special;
  with_special.add_special_tokens = true;
  const auto encode = (*tokenizer)->Encode("hello", with_special);
  ASSERT_FALSE(encode.ok());
  EXPECT_EQ(encode.status().code(), absl::StatusCode::kUnimplemented);

  DecodeOptions skip_special;
  skip_special.skip_special_tokens = true;
  const auto decode = (*tokenizer)->Decode({1, 2, 3}, skip_special);
  ASSERT_FALSE(decode.ok());
  EXPECT_EQ(decode.status().code(), absl::StatusCode::kUnimplemented);

  // Compatibility defaults go through the plain path and agree with it.
  const auto plain_encode = (*tokenizer)->Encode("hello");
  ASSERT_TRUE(plain_encode.ok()) << plain_encode.status();
  const auto optioned_encode = (*tokenizer)->Encode("hello", EncodeOptions{});
  ASSERT_TRUE(optioned_encode.ok()) << optioned_encode.status();
  EXPECT_EQ(*optioned_encode, *plain_encode);

  const auto plain_decode = (*tokenizer)->Decode(*plain_encode);
  ASSERT_TRUE(plain_decode.ok()) << plain_decode.status();
  const auto optioned_decode =
      (*tokenizer)->Decode(*plain_encode, DecodeOptions{});
  ASSERT_TRUE(optioned_decode.ok()) << optioned_decode.status();
  EXPECT_EQ(*optioned_decode, *plain_decode);
}

TEST(TokenizerTest, EncodeBatchMatchesPerItem) {
  auto tokenizer = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  const std::vector<std::string> texts = {"one", "hello world", "héllo 世界", "aaaa"};
  const auto batched = (*tokenizer)->EncodeBatch(texts, EncodeOptions{});
  ASSERT_TRUE(batched.ok()) << batched.status();
  ASSERT_EQ(batched->size(), texts.size());
  for (std::size_t i = 0; i < texts.size(); ++i) {
    const auto single = (*tokenizer)->Encode(texts[i]);
    ASSERT_TRUE(single.ok()) << single.status();
    EXPECT_EQ((*batched)[i], *single) << "batch item " << i << " diverged";
  }
}

TEST(TokenizerTest, MissingFileFailsToLoad) {
  const auto missing = Tokenizer::FromFile("tests/testdata/tiny_tokenizer/missing.json");
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace
}  // namespace inferx
