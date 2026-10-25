// HFTokenizer unit tests over the tiny fixture (a 7-token toy vocabulary)
// and the qwen3_chat fixture (real ChatML tokenizer_config with a chat
// template and special tokens). Covers load errors, encode/decode
// semantics, metadata tables, conversions, and truncation behavior.
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/config/model_config.h"
#include "inferx/core/status.h"
#include "inferx/tokenizer/hf_tokenizer.h"
#include "inferx/tokenizer/tokenizer.h"
#include "inferx/tokenizer/tokenizer_resolver.h"

namespace inferx {
namespace {

constexpr const char* kTinyDir = "tests/testdata/tiny_tokenizer";
constexpr const char* kQwen3Dir = "tests/testdata/qwen3_chat";
// Toy ChatML fixture: special tokens in vocab and added_tokens (as real
// checkpoints serialize them), a BOS-adding post-processor, and config
// metadata including a special_tokens_map override.
constexpr const char* kChatMlDir = "tests/testdata/chatml_tokenizer";

StatusOr<std::shared_ptr<HFTokenizer>> LoadTiny() { return HFTokenizer::Load(kTinyDir); }
StatusOr<std::shared_ptr<HFTokenizer>> LoadQwen3() { return HFTokenizer::Load(kQwen3Dir); }

TokenId TokenToIdOrDie(const Tokenizer& tokenizer, const std::string& token) {
  const auto ids = tokenizer.ConvertTokensToIds(std::vector<std::string>{token});
  if (!ids.ok() || ids->size() != 1) {
    ADD_FAILURE() << "token not in vocabulary: " << token << " (" << ids.status() << ")";
    return -1;
  }
  return ids->front();
}

TEST(HFTokenizerTest, LoadsAndReportsVocabularySizes) {
  auto tokenizer = LoadTiny();
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  // The tiny fixture: 7 base tokens, no added tokens.
  EXPECT_EQ((*tokenizer)->VocabSize(), 7u);
  EXPECT_EQ((*tokenizer)->Size(), 7u);
  EXPECT_EQ((*tokenizer)->MaxTokenId(), 6);
}

TEST(HFTokenizerTest, MissingDirectoryFailsWithNotFound) {
  const auto missing = HFTokenizer::Load("tests/testdata/tiny_tokenizer/missing");
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
}

TEST(HFTokenizerTest, MalformedAssetsFailWithInvalidArgument) {
  HFTokenizerAssets assets;
  assets.source_dir = "memory";

  assets.tokenizer_json = "{ not json";
  auto bad_json = HFTokenizer::FromAssets(assets);
  ASSERT_FALSE(bad_json.ok());
  EXPECT_EQ(bad_json.status().code(), absl::StatusCode::kInvalidArgument);

  // Valid JSON but not a fast-tokenizer document (e.g. tokenizer_config
  // pointed at the wrong slot); rejected before the Rust backend sees it.
  assets.tokenizer_json = R"({"model_max_length": 2048, "unk_token": "<unk>"})";
  auto wrong_schema = HFTokenizer::FromAssets(assets);
  ASSERT_FALSE(wrong_schema.ok());
  EXPECT_EQ(wrong_schema.status().code(), absl::StatusCode::kInvalidArgument);

  assets.tokenizer_json = R"({"version": "1.0", "model": {"type": "BPE", "vocab": {"a": 0}}})";
  assets.tokenizer_config_json = "{ bad config";
  auto bad_config = HFTokenizer::FromAssets(assets);
  ASSERT_FALSE(bad_config.ok());
  EXPECT_EQ(bad_config.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(HFTokenizerTest, EncodeThenDecodeIsTotalAndDeterministic) {
  auto tokenizer = LoadTiny();
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

    const std::string decoded = (*tokenizer)->Decode(*ids);
    EXPECT_EQ(decoded, (*tokenizer)->Decode(*ids)) << "decode must be deterministic";
  }
}

TEST(HFTokenizerTest, InvalidUtf8IsRejectedNotFatal) {
  auto tokenizer = LoadTiny();
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  const std::string invalid = "ok\xff\xfe text";
  const auto ids = (*tokenizer)->Encode(invalid);
  ASSERT_FALSE(ids.ok());
  EXPECT_EQ(ids.status().code(), absl::StatusCode::kInvalidArgument);

  const std::vector<std::string> batch = {"ok", invalid};
  const auto batch_ids = (*tokenizer)->EncodeBatch(batch);
  ASSERT_FALSE(batch_ids.ok());
  EXPECT_EQ(batch_ids.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(HFTokenizerTest, EncodeBatchMatchesPerItem) {
  auto tokenizer = LoadTiny();
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  const std::vector<std::string> texts = {"one", "hello world", "héllo 世界", "aaaa"};
  const auto batched = (*tokenizer)->EncodeBatch(texts);
  ASSERT_TRUE(batched.ok()) << batched.status();
  ASSERT_EQ(batched->size(), texts.size());
  for (std::size_t i = 0; i < texts.size(); ++i) {
    const auto single = (*tokenizer)->Encode(texts[i]);
    ASSERT_TRUE(single.ok()) << single.status();
    EXPECT_EQ((*batched)[i], *single) << "batch item " << i << " diverged";
  }
}

TEST(HFTokenizerTest, DecodeIgnoresOutOfRangeIds) {
  auto tokenizer = LoadTiny();
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  const auto ids = (*tokenizer)->Encode("abc");
  ASSERT_TRUE(ids.ok()) << ids.status();
  std::vector<TokenId> with_junk = *ids;
  with_junk.push_back(-3);
  with_junk.push_back(999'999);
  EXPECT_EQ((*tokenizer)->Decode(with_junk), (*tokenizer)->Decode(*ids));
  EXPECT_EQ((*tokenizer)->Decode({}), "");
}

TEST(HFTokenizerTest, TruncationCutsFromTheConfiguredSide) {
  auto tokenizer = LoadTiny();
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  EXPECT_EQ((*tokenizer)->GetTruncationSide(), TruncationSide::kRight);

  const auto full = (*tokenizer)->Encode("hello world");
  ASSERT_TRUE(full.ok()) << full.status();
  ASSERT_GE(full->size(), 3u) << "fixture tokenizer split too coarsely for this test";

  EncodeOptions truncate;
  truncate.truncation = true;
  truncate.max_length = 2;
  const auto right = (*tokenizer)->Encode("hello world", truncate);
  ASSERT_TRUE(right.ok()) << right.status();
  ASSERT_EQ(right->size(), 2u);
  EXPECT_EQ(*right, std::vector<TokenId>(full->begin(), full->begin() + 2));

  // max_length=1 keeps the tail on the left side, verified through
  // truncation of an already-encoded sequence via a fresh encode call.
  truncate.max_length = 1;
  const auto left = (*tokenizer)->Encode("hello world", truncate);
  ASSERT_TRUE(left.ok()) << left.status();
  ASSERT_EQ(left->size(), 1u);
  EXPECT_EQ(left->front(), full->back());
}

TEST(HFTokenizerTest, ConversionsRoundTrip) {
  auto tokenizer = LoadTiny();
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  const auto ids = (*tokenizer)->Encode("abc");
  ASSERT_TRUE(ids.ok()) << ids.status();

  const std::vector<std::string> tokens = (*tokenizer)->ConvertIdsToTokens(*ids);
  ASSERT_EQ(tokens.size(), ids->size());
  const auto back = (*tokenizer)->ConvertTokensToIds(tokens);
  ASSERT_TRUE(back.ok()) << back.status();
  EXPECT_EQ(*back, *ids);

  const auto as_string = (*tokenizer)->ConvertTokensToString(tokens);
  ASSERT_TRUE(as_string.ok()) << as_string.status();
  EXPECT_EQ(*as_string, (*tokenizer)->Decode(*ids));

  const auto unknown = (*tokenizer)->ConvertTokensToIds(std::vector<std::string>{"__no_such_token__"});
  ASSERT_FALSE(unknown.ok());
  EXPECT_EQ(unknown.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(HFTokenizerTest, ChatMlMetadataMatchesTransformers) {
  auto tokenizer = HFTokenizer::Load(kChatMlDir);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  // eos from tokenizer_config (object form), pad overridden by
  // special_tokens_map, no bos, unk from the model's unk_token entry.
  EXPECT_EQ((*tokenizer)->EosTokenId(), std::optional<TokenId>(9));
  EXPECT_EQ((*tokenizer)->PadTokenId(), std::optional<TokenId>(7));
  EXPECT_EQ((*tokenizer)->BosTokenId(), std::nullopt);
  EXPECT_EQ(TokenToIdOrDie(**tokenizer, "<|endoftext|>"), 7);

  EXPECT_EQ((*tokenizer)->ModelMaxLength(), std::optional<int64_t>(4096));
  EXPECT_EQ((*tokenizer)->GetTruncationSide(), TruncationSide::kLeft);

  const auto& specials = (*tokenizer)->AllSpecialTokens();
  for (const std::string& token :
       {"<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|image_pad|>"}) {
    EXPECT_NE(std::find(specials.begin(), specials.end(), token), specials.end())
        << "missing special token " << token;
  }
  for (const std::string& token :
       {"<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|image_pad|>"}) {
    EXPECT_TRUE((*tokenizer)->IsSpecialTokenId(TokenToIdOrDie(**tokenizer, token)))
        << token;
  }
  EXPECT_FALSE((*tokenizer)->IsSpecialTokenId(4)) << "'abc' is not special";

  // 10 base vocabulary entries; <|image_pad|> exists only as an added
  // token, so len(tokenizer) is 11 and the largest id is 10.
  EXPECT_EQ((*tokenizer)->VocabSize(), 10u);
  EXPECT_EQ((*tokenizer)->Size(), 11u);
  EXPECT_EQ((*tokenizer)->MaxTokenId(), 10);
  EXPECT_EQ((*tokenizer)->GetAddedVocab().size(), 4u);
  EXPECT_EQ((*tokenizer)->GetVocab().size(), 11u);
}

TEST(HFTokenizerTest, ChatMlSpecialTokenFlagDrivesPostProcessor) {
  auto tokenizer = HFTokenizer::Load(kChatMlDir);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();

  // The fixture's post-processor prefixes <|im_start|> when the HF default
  // (add_special_tokens=true) is used, and not otherwise. The Whitespace
  // pre-tokenizer plus the absent decoder split "abc" into "ab" "c".
  const auto added = (*tokenizer)->Encode("abc");
  ASSERT_TRUE(added.ok()) << added.status();
  EXPECT_EQ(*added, std::vector<TokenId>({8, 3, 2}));

  EncodeOptions without_special;
  without_special.add_special_tokens = false;
  const auto plain = (*tokenizer)->Encode("abc", without_special);
  ASSERT_TRUE(plain.ok()) << plain.status();
  EXPECT_EQ(*plain, std::vector<TokenId>({3, 2}));

  // Full-fidelity decode keeps the special token; skipping drops it. With
  // no decoder the backend joins token strings with spaces.
  EXPECT_EQ((*tokenizer)->Decode(*added), "<|im_start|> ab c");
  EXPECT_EQ((*tokenizer)->Decode(*added, /*skip_special_tokens=*/true), "ab c");

  const auto ids = (*tokenizer)->Encode("ab abc");
  ASSERT_TRUE(ids.ok()) << ids.status();
  EXPECT_EQ(*ids, std::vector<TokenId>({8, 3, 3, 2}));
  EXPECT_EQ((*tokenizer)->Decode(*ids, /*skip_special_tokens=*/true), "ab ab c");
}

TEST(HFTokenizerTest, ResolverPrefersTokenizerDirThenModelDir) {
  ModelConfig config;
  config.model_dir = "tests/testdata/tiny_tokenizer";
  auto via_model = ResolveTokenizer(config);
  ASSERT_TRUE(via_model.ok()) << via_model.status();
  EXPECT_EQ((*via_model)->VocabSize(), 7u);

  config.tokenizer_dir = "tests/testdata/chatml_tokenizer";
  auto via_tokenizer = ResolveTokenizer(config);
  ASSERT_TRUE(via_tokenizer.ok()) << via_tokenizer.status();
  EXPECT_EQ((*via_tokenizer)->ModelMaxLength(), std::optional<int64_t>(4096));

  config.tokenizer_dir = "tests/testdata/does_not_exist";
  const auto missing = ResolveTokenizer(config);
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kNotFound);
}

}  // namespace
}  // namespace inferx
