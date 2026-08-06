#include <gtest/gtest.h>
#include <tokenizers_cpp.h>

#include <atomic>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr char kTokenizerJson[] = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [],
  "normalizer": null,
  "pre_tokenizer": {"type": "Whitespace"},
  "post_processor": null,
  "decoder": null,
  "model": {
    "type": "WordLevel",
    "vocab": {"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3,
              "delta": 4, "epsilon": 5},
    "unk_token": "[UNK]"
  }
})json";

std::unique_ptr<tokenizers::Tokenizer> MakeTokenizer() {
  return tokenizers::Tokenizer::FromBlobJSON(kTokenizerJson);
}

std::string ReadFile(const std::string& path) {
  std::ifstream input(path, std::ios::binary);
  std::ostringstream contents;
  contents << input.rdbuf();
  return contents.str();
}

TEST(TokenizersCppMetadata, ResolvesHuggingFaceSpecialTokens) {
  constexpr char kConfig[] = R"json({
    "bos_token": "alpha",
    "eos_token": {"content": "epsilon"},
    "pad_token": "[UNK]"
  })json";
  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      tokenizers::Tokenizer::FromBlobJSON(kTokenizerJson, kConfig);

  ASSERT_NE(tokenizer, nullptr);
  EXPECT_EQ(tokenizer->GetBosTokenId(), 1);
  EXPECT_EQ(tokenizer->GetEosTokenId(), 5);
  EXPECT_EQ(tokenizer->GetPadTokenId(), 0);
}

TEST(TokenizersCppMetadata, LoadsDownloadedHuggingFaceCheckpoint) {
  const char* directory = std::getenv("INFERX_HF_TOKENIZER_TEST_DIR");
  if (directory == nullptr) {
    GTEST_SKIP()
        << "set INFERX_HF_TOKENIZER_TEST_DIR to a downloaded checkpoint";
  }
  const std::string tokenizer_json =
      ReadFile(std::string(directory) + "/tokenizer.json");
  const std::string tokenizer_config =
      ReadFile(std::string(directory) + "/tokenizer_config.json");
  ASSERT_FALSE(tokenizer_json.empty());
  ASSERT_FALSE(tokenizer_config.empty());

  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      tokenizers::Tokenizer::FromBlobJSON(tokenizer_json, tokenizer_config);
  ASSERT_NE(tokenizer, nullptr);
  EXPECT_EQ(tokenizer->GetBosTokenId(), -1);
  EXPECT_EQ(tokenizer->GetEosTokenId(), 151645);
  EXPECT_EQ(tokenizer->GetPadTokenId(), 151643);
  EXPECT_EQ(tokenizer->Decode(tokenizer->Encode("Hello, tokenizer!")),
            "Hello, tokenizer!");
}

TEST(TokenizersCppMetadata, LoadsDownloadedDeepSeekV2Checkpoint) {
  const char* directory =
      std::getenv("INFERX_HF_DEEPSEEK_V2_TOKENIZER_TEST_DIR");
  if (directory == nullptr) {
    GTEST_SKIP()
        << "set INFERX_HF_DEEPSEEK_V2_TOKENIZER_TEST_DIR to the checkpoint";
  }
  const std::string tokenizer_json =
      ReadFile(std::string(directory) + "/tokenizer.json");
  const std::string tokenizer_config =
      ReadFile(std::string(directory) + "/tokenizer_config.json");
  ASSERT_FALSE(tokenizer_json.empty());
  ASSERT_FALSE(tokenizer_config.empty());

  std::unique_ptr<tokenizers::Tokenizer> tokenizer =
      tokenizers::Tokenizer::FromBlobJSON(tokenizer_json, tokenizer_config);
  ASSERT_NE(tokenizer, nullptr);
  EXPECT_EQ(tokenizer->GetBosTokenId(), 100000);
  EXPECT_EQ(tokenizer->GetEosTokenId(), 100001);
  EXPECT_EQ(tokenizer->GetPadTokenId(), 100001);
  EXPECT_EQ(tokenizer->Decode(tokenizer->Encode("Hello, DeepSeek-V2!")),
            "Hello, DeepSeek-V2!");
}

// This is the supported ownership pattern when callers cannot rely on a
// backend thread-safety guarantee: every thread owns its handle.
TEST(TokenizersCppThreadSafety, IndependentHandlesEncodeAndDecodeConcurrently) {
  constexpr int kThreadCount = 8;
  constexpr int kIterations = 2000;
  std::atomic<bool> start = false;
  std::atomic<int> failures = 0;
  std::vector<std::thread> threads;

  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      std::unique_ptr<tokenizers::Tokenizer> tokenizer = MakeTokenizer();
      const std::string text =
          thread_index % 2 == 0 ? "alpha beta gamma" : "delta epsilon";
      const std::vector<int32_t> expected_ids =
          thread_index % 2 == 0 ? std::vector<int32_t>{1, 2, 3}
                                : std::vector<int32_t>{4, 5};

      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int iteration = 0; iteration < kIterations; ++iteration) {
        const std::vector<int32_t> ids = tokenizer->Encode(text);
        if (ids != expected_ids || tokenizer->Decode(ids) != text) {
          failures.fetch_add(1, std::memory_order_relaxed);
          return;
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) thread.join();

  EXPECT_EQ(failures.load(), 0);
}

// tokenizers-cpp's Hugging Face bridge is not safe for this pattern today.
// Its Rust FFI mutably borrows the common handle, and Decode stores its result
// in a handle-owned string that is fetched by a second FFI call. Keep this
// regression probe disabled in ordinary builds because exercising undefined
// behavior is unsuitable for CI. Run it under ThreadSanitizer when evaluating
// a new tokenizers-cpp revision:
//
//   ./tokenizers_cpp_thread_safety_test \
//     --gtest_also_run_disabled_tests \
//     --gtest_filter='*SharedHandle*'
TEST(TokenizersCppThreadSafety,
     DISABLED_SharedHandleEncodeAndDecodeConcurrently) {
  constexpr int kThreadCount = 8;
  constexpr int kIterations = 10000;
  std::unique_ptr<tokenizers::Tokenizer> tokenizer = MakeTokenizer();
  std::atomic<bool> start = false;
  std::atomic<int> failures = 0;
  std::vector<std::thread> threads;

  for (int thread_index = 0; thread_index < kThreadCount; ++thread_index) {
    threads.emplace_back([&, thread_index] {
      const std::string text =
          thread_index % 2 == 0 ? "alpha beta gamma" : "delta epsilon";
      const std::vector<int32_t> expected_ids =
          thread_index % 2 == 0 ? std::vector<int32_t>{1, 2, 3}
                                : std::vector<int32_t>{4, 5};

      while (!start.load(std::memory_order_acquire)) {
        std::this_thread::yield();
      }

      for (int iteration = 0; iteration < kIterations; ++iteration) {
        const std::vector<int32_t> ids = tokenizer->Encode(text);
        if (ids != expected_ids || tokenizer->Decode(ids) != text) {
          failures.fetch_add(1, std::memory_order_relaxed);
        }
      }
    });
  }

  start.store(true, std::memory_order_release);
  for (std::thread& thread : threads) thread.join();

  EXPECT_EQ(failures.load(), 0)
      << "the shared tokenizers-cpp handle corrupted concurrent results";
}

}  // namespace
