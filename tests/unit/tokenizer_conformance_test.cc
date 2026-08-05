/// Conformance of our byte-level BPE against HuggingFace's tokenizer.
///
/// ARCHITECTURE.md R7 wanted exactness by construction, through FFI to the Rust
/// implementation. We reimplemented instead, so exactness is a claim rather
/// than a guarantee -- and this file is the evidence for the claim. Every string
/// in `testdata/tokenizer_corpus.txt` carries the ids HuggingFace produced for
/// it, and we require exact equality.
///
/// There is deliberately no tolerance here and there never should be. Token ids
/// are discrete: an "almost right" tokenization is a different prompt, and a
/// model given a different prompt produces different output for reasons that
/// are invisible downstream. If this test fails, the tokenizer is wrong.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/tokenizer/chat_template.h"
#include "inferx/tokenizer/tokenizer.h"
#include "inferx/tokenizer/unicode.h"

namespace inferx::tokenizer {
namespace {

// Kept in sync with scripts/gen_tokenizer_corpus.py, which writes the corpus.
constexpr std::string_view kCorpusPath = "testdata/tokenizer_corpus.txt";

// The o200k corpus, generated the same way from gpt-oss's tokenizer.json.
// Separate file so one corpus can be regenerated without touching the other.
constexpr std::string_view kO200kCorpusPath =
    "testdata/tokenizer_corpus_o200k.txt";

std::string Base64Decode(std::string_view text) {
  static constexpr std::string_view kAlphabet =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

  std::string out;
  uint32_t buffer = 0;
  int bits = 0;

  for (const char c : text) {
    if (c == '=') break;

    const size_t index = kAlphabet.find(c);
    if (index == std::string_view::npos) continue;

    buffer = (buffer << 6) | static_cast<uint32_t>(index);
    bits += 6;

    if (bits >= 8) {
      bits -= 8;
      out.push_back(static_cast<char>((buffer >> bits) & 0xFF));
    }
  }

  return out;
}

struct Record {
  std::string text;
  std::vector<int32_t> ids;
  int line = 0;
};

// Resolves the corpus and checkpoint relative to the source tree, so the test
// runs the same from the build directory or from the repository root.
std::string RepoPath(std::string_view relative) {
  for (const std::string_view prefix : {"", "../", "../../"}) {
    std::string candidate = std::string(prefix) + std::string(relative);

    if (std::ifstream(candidate).good()) return candidate;
  }

  return std::string(relative);
}

std::vector<Record> LoadCorpusAt(const std::string& path) {
  std::ifstream in(path);

  std::vector<Record> records;
  std::string line;
  int number = 0;

  while (std::getline(in, line)) {
    ++number;

    if (line.empty() || line[0] == '#') continue;

    const size_t tab = line.find('\t');
    if (tab == std::string::npos) continue;

    Record record;
    record.line = number;
    record.text = Base64Decode(line.substr(0, tab));

    std::istringstream ids(line.substr(tab + 1));
    int32_t id = 0;

    while (ids >> id) record.ids.push_back(id);

    records.push_back(std::move(record));
  }

  return records;
}

std::vector<Record> LoadCorpus() {
  return LoadCorpusAt(RepoPath(kCorpusPath));
}

// The Qwen2 conformance suite is parametric over (checkpoint, corpus) so the
// o200k suite below can share the loader. Each (checkpoint, corpus) pair is
// its own fixture: the vocabularies differ, so a single tokenizer_ cannot
// serve both.
std::string Qwen2CheckpointDir() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return {};

  const std::string base = std::string(home) +
      "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots";

  if (std::FILE* pipe = popen(("ls -d " + base + "/*/ 2>/dev/null").c_str(), "r");
      pipe != nullptr) {
    char buffer[4096] = {};
    std::string path;

    if (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      path = buffer;
      if (!path.empty() && path.back() == '\n') path.pop_back();
      if (!path.empty() && path.back() == '/') path.pop_back();
    }

    pclose(pipe);
    return path;
  }

  return {};
}

std::string O200kCheckpointDir() {
  const char* home = std::getenv("HOME");
  if (home == nullptr) return {};

  const std::string base = std::string(home) +
      "/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots";

  if (std::FILE* pipe = popen(("ls -d " + base + "/*/ 2>/dev/null").c_str(), "r");
      pipe != nullptr) {
    char buffer[4096] = {};
    std::string path;

    if (std::fgets(buffer, sizeof(buffer), pipe) != nullptr) {
      path = buffer;
      if (!path.empty() && path.back() == '\n') path.pop_back();
      if (!path.empty() && path.back() == '/') path.pop_back();
    }

    pclose(pipe);
    return path;
  }

  return {};
}

class TokenizerConformance : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string dir = Qwen2CheckpointDir();

    if (dir.empty()) return;

    StatusOr<std::unique_ptr<Tokenizer>> loaded =
        Tokenizer::LoadFromDirectory(dir);

    ASSERT_TRUE(loaded.ok())
        << "the checkpoint is present but its tokenizer would not load: "
        << loaded.status().ToString();

    tokenizer_ = loaded->release();
  }

  static void TearDownTestSuite() {
    delete tokenizer_;
    tokenizer_ = nullptr;
  }

  void SetUp() override {
    if (tokenizer_ == nullptr) {
      GTEST_SKIP() << "Qwen2.5-3B-Instruct checkpoint not present";
    }
  }

  static Tokenizer* tokenizer_;
};

Tokenizer* TokenizerConformance::tokenizer_ = nullptr;

TEST_F(TokenizerConformance, CorpusIsPresentAndNonTrivial) {
  const std::vector<Record> records = LoadCorpus();

  // A missing or truncated corpus would make every other test in this file
  // pass vacuously, which is the failure mode most likely to go unnoticed.
  ASSERT_GE(records.size(), 100u)
      << "corpus has " << records.size()
      << " records; regenerate with scripts/gen_tokenizer_corpus.py";
}

TEST_F(TokenizerConformance, MatchesHuggingFaceOnEveryCorpusEntry) {
  const std::vector<Record> records = LoadCorpus();
  ASSERT_FALSE(records.empty());

  int mismatches = 0;

  for (const Record& record : records) {
    const std::vector<int32_t> actual = tokenizer_->Encode(record.text);

    if (actual == record.ids) continue;

    ++mismatches;

    // Report every mismatch rather than stopping at the first: when a rule is
    // wrong it usually breaks a whole class of input, and the shape of the
    // failure set says which rule it is.
    if (mismatches <= 20) {
      std::ostringstream expected;
      std::ostringstream got;

      for (const int32_t id : record.ids) {
        expected << id << "(" << tokenizer_->IdToToken(id) << ") ";
      }
      for (const int32_t id : actual) {
        got << id << "(" << tokenizer_->IdToToken(id) << ") ";
      }

      ADD_FAILURE() << "corpus line " << record.line << "\n"
                    << "  input    : " << ::testing::PrintToString(record.text)
                    << "\n  expected : " << expected.str()
                    << "\n  actual   : " << got.str();
    }
  }

  EXPECT_EQ(mismatches, 0) << mismatches << " of " << records.size()
                           << " corpus entries tokenize differently from "
                              "HuggingFace";
}

TEST_F(TokenizerConformance, DecodeRoundTripsEveryCorpusEntry) {
  // Decoding the *expected* ids must reproduce the input text, up to NFC --
  // normalization is lossy by design, so a decomposed input comes back
  // composed.
  for (const Record& record : LoadCorpus()) {
    const std::string decoded = tokenizer_->Decode(record.ids);

    EXPECT_EQ(decoded, unicode::NormalizeNfc(record.text))
        << "corpus line " << record.line;
  }
}

TEST_F(TokenizerConformance, IncrementalDecodeMatchesBatchDecode) {
  // The streaming path must produce exactly the bytes the batch path produces.
  // If it does not, a completion read over SSE differs from the same completion
  // read from the non-streaming endpoint -- which is the kind of discrepancy
  // that gets reported as a model bug.
  for (const Record& record : LoadCorpus()) {
    if (record.ids.empty()) continue;

    IncrementalDecoder decoder(tokenizer_, /*skip_special=*/false);

    std::string streamed;
    for (const int32_t id : record.ids) streamed += decoder.Push(id);
    streamed += decoder.Flush();

    EXPECT_EQ(streamed, tokenizer_->Decode(record.ids))
        << "corpus line " << record.line;
  }
}

TEST_F(TokenizerConformance, IncrementalDecodeNeverSplitsACharacter) {
  // Every prefix the stream emits must be valid UTF-8 on its own. A client
  // rendering chunks as they arrive shows a replacement character otherwise.
  const std::vector<int32_t> ids = tokenizer_->Encode("你好，世界！🙂👍🏽");

  IncrementalDecoder decoder(tokenizer_, /*skip_special=*/false);

  for (const int32_t id : ids) {
    const std::string chunk = decoder.Push(id);

    EXPECT_FALSE(unicode::EndsWithPartialUtf8(chunk))
        << "chunk ends mid-character: " << ::testing::PrintToString(chunk);
  }
}

TEST_F(TokenizerConformance, EncodeOrdinaryDoesNotHonourControlTokens) {
  // User text must not be able to inject a turn boundary. This is the reason
  // the two entry points exist at all.
  const std::vector<int32_t> as_control = tokenizer_->Encode("<|im_end|>");
  const std::vector<int32_t> as_text = tokenizer_->EncodeOrdinary("<|im_end|>");

  ASSERT_EQ(as_control.size(), 1u);
  EXPECT_EQ(as_control[0], tokenizer_->TokenToId("<|im_end|>").value());

  EXPECT_GT(as_text.size(), 1u)
      << "user-supplied text was tokenized as a control token";
  EXPECT_EQ(std::count(as_text.begin(), as_text.end(), as_control[0]), 0);
}

TEST_F(TokenizerConformance, EosIsImEnd) {
  // Qwen2.5-Instruct stops on <|im_end|>, not <|endoftext|>; getting this wrong
  // makes generation run to the token limit on every request.
  EXPECT_EQ(tokenizer_->eos_id(), tokenizer_->TokenToId("<|im_end|>").value());
}

TEST_F(TokenizerConformance, ChatTemplateMatchesTheCheckpointsTemplate) {
  // The expected strings here are what the checkpoint's Jinja template renders;
  // see scripts/gen_tokenizer_corpus.py's sibling check in the test below.
  const StatusOr<std::string> rendered = ApplyQwen2ChatTemplate(
      {{"user", "What is 2+2?"}}, /*add_generation_prompt=*/true);

  ASSERT_TRUE(rendered.ok()) << rendered.status().ToString();

  EXPECT_EQ(*rendered,
            "<|im_start|>system\n"
            "You are Qwen, created by Alibaba Cloud. You are a helpful "
            "assistant.<|im_end|>\n"
            "<|im_start|>user\nWhat is 2+2?<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST_F(TokenizerConformance, ChatTemplateUsesACallerSuppliedSystemTurn) {
  const StatusOr<std::string> rendered = ApplyQwen2ChatTemplate(
      {{"system", "Be terse."}, {"user", "Hi"}, {"assistant", "Hello."},
       {"user", "Again"}},
      /*add_generation_prompt=*/true);

  ASSERT_TRUE(rendered.ok()) << rendered.status().ToString();

  EXPECT_EQ(*rendered,
            "<|im_start|>system\nBe terse.<|im_end|>\n"
            "<|im_start|>user\nHi<|im_end|>\n"
            "<|im_start|>assistant\nHello.<|im_end|>\n"
            "<|im_start|>user\nAgain<|im_end|>\n"
            "<|im_start|>assistant\n");
}

TEST_F(TokenizerConformance, ChatTemplateRejectsRolesItCannotRender) {
  EXPECT_FALSE(
      ApplyQwen2ChatTemplate({{"tool", "{}"}}, true).ok())
      << "a tool turn was silently dropped rather than rejected";
}

TEST_F(TokenizerConformance, LongWhitespaceRunIsNotQuadratic) {
  // The BPE merge loop is a priority queue precisely so that this cannot become
  // a denial-of-service vector: a pre-tokenized piece is unbounded, and a run
  // of spaces arrives as a single word. 200k spaces would take minutes under a
  // rescan-for-minimum implementation.
  const std::string input(200000, ' ');

  const auto start = std::chrono::steady_clock::now();
  const std::vector<int32_t> ids = tokenizer_->Encode(input);
  const auto elapsed = std::chrono::steady_clock::now() - start;

  EXPECT_FALSE(ids.empty());
  EXPECT_LT(std::chrono::duration_cast<std::chrono::seconds>(elapsed).count(), 5)
      << "encoding a long whitespace run took "
      << std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count()
      << " ms, which suggests the merge loop is quadratic";
}

// --- o200k (gpt-oss) -------------------------------------------------------
//
// The o200k pattern is the second split this tokenizer implements, and its
// conformance bar is the same as Qwen2's: exact id equality on every corpus
// entry, because "almost right" is a different prompt. The corpus is generated
// from gpt-oss's own tokenizer.json by the same script, and is skipped rather
// than failed when the checkpoint is absent.

class O200kConformance : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    const std::string dir = O200kCheckpointDir();
    if (dir.empty()) return;

    StatusOr<std::unique_ptr<Tokenizer>> loaded =
        Tokenizer::LoadFromDirectory(dir);

    ASSERT_TRUE(loaded.ok())
        << "gpt-oss checkpoint is present but its tokenizer would not load: "
        << loaded.status().ToString();

    tokenizer_ = loaded->release();
  }

  static void TearDownTestSuite() {
    delete tokenizer_;
    tokenizer_ = nullptr;
  }

  void SetUp() override {
    if (tokenizer_ == nullptr) {
      GTEST_SKIP() << "gpt-oss-20b checkpoint not present";
    }
  }

  static Tokenizer* tokenizer_;
};

Tokenizer* O200kConformance::tokenizer_ = nullptr;

TEST_F(O200kConformance, LoadsAndPicksTheO200kPattern) {
  // A sanity check the Qwen2 suite has implicitly via its corpus: the loaded
  // tokenizer must actually be running the o200k pattern, not silently falling
  // back to Qwen2's. The distinguishing case is CamelCase, which the corpus
  // tests below cover -- but a separate assertion here names the failure mode.
  const std::vector<int32_t> hello_world = tokenizer_->Encode("HelloWorld");
  ASSERT_GT(hello_world.size(), 1u)
      << "'HelloWorld' encoded as a single token, which means the o200k "
         "CamelCase splitter is not running";
}

TEST_F(O200kConformance, CorpusIsPresentAndNonTrivial) {
  const std::vector<Record> records =
      LoadCorpusAt(RepoPath(kO200kCorpusPath));

  ASSERT_GE(records.size(), 100u)
      << "o200k corpus has " << records.size()
      << " records; regenerate with scripts/gen_tokenizer_corpus.py pointed "
         "at the gpt-oss checkpoint";
}

TEST_F(O200kConformance, MatchesHuggingFaceOnEveryCorpusEntry) {
  const std::vector<Record> records =
      LoadCorpusAt(RepoPath(kO200kCorpusPath));
  ASSERT_FALSE(records.empty());

  int mismatches = 0;

  for (const Record& record : records) {
    const std::vector<int32_t> actual = tokenizer_->Encode(record.text);

    if (actual == record.ids) continue;

    ++mismatches;

    if (mismatches <= 20) {
      std::ostringstream expected;
      std::ostringstream got;

      for (const int32_t id : record.ids) {
        expected << id << "(" << tokenizer_->IdToToken(id) << ") ";
      }
      for (const int32_t id : actual) {
        got << id << "(" << tokenizer_->IdToToken(id) << ") ";
      }

      ADD_FAILURE() << "o200k corpus line " << record.line << "\n"
                    << "  input    : " << ::testing::PrintToString(record.text)
                    << "\n  expected : " << expected.str()
                    << "\n  actual   : " << got.str();
    }
  }

  EXPECT_EQ(mismatches, 0) << mismatches << " of " << records.size()
                           << " o200k corpus entries tokenize differently "
                              "from HuggingFace";
}

}  // namespace
}  // namespace inferx::tokenizer
