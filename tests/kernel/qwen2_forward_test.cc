// Runs the real Qwen2.5-3B stack end to end.
//
// There is no HF reference on this machine yet, so this does not claim "logits
// match HF" -- that is the last piece of M2 and is deliberately not asserted
// here. What it does check is everything that can be checked without one, and
// the list is longer than it looks: a forward pass with a transposed
// projection, a mis-indexed RoPE, a broken causal mask or a wrong norm epsilon
// produces logits that are finite and plausible-looking, but it does not
// produce a sharply peaked distribution that ranks sensible continuations
// first. Several of the assertions below fail loudly for each of those.

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/model/qwen2.h"

namespace inferx::model {
namespace {

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

class Qwen2ForwardTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!CudaAvailable()) return;

    auto loaded = Qwen2Model::LoadFromDirectory(CheckpointDir());
    if (!loaded.ok()) {
      load_status_ = loaded.status();
      return;
    }

    model_ = new Qwen2Model(*std::move(loaded));
  }

  static void TearDownTestSuite() {
    delete model_;
    model_ = nullptr;
  }

  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
    if (model_ == nullptr) GTEST_SKIP() << "model not loaded: " << load_status_;
  }

  // Loaded once: 6.2 GB over PCIe per test would dominate everything.
  static Qwen2Model* model_;
  static Status load_status_;
};

Qwen2Model* Qwen2ForwardTest::model_ = nullptr;
Status Qwen2ForwardTest::load_status_ = OkStatus();

TEST_F(Qwen2ForwardTest, LoadsTheExpectedAmountOfWeight) {
  EXPECT_EQ(model_->config().num_hidden_layers, 36);
  EXPECT_EQ(model_->config().hidden_size, 2048);
  EXPECT_TRUE(model_->config().tie_word_embeddings);

  // Tied embeddings mean the LM head is not a second copy, so the resident
  // weight is the checkpoint's own size rather than that plus 600 MB.
  EXPECT_EQ(model_->WeightBytes(), 6171877376u);
}

TEST_F(Qwen2ForwardTest, ProducesFiniteLogits) {
  std::vector<float> logits;
  ASSERT_TRUE(model_->ForwardLastLogits({100, 200, 300, 400}, &logits).ok());

  ASSERT_EQ(logits.size(), 151936u);

  for (size_t i = 0; i < logits.size(); ++i) {
    ASSERT_TRUE(std::isfinite(logits[i]))
        << "logit " << i << " is " << logits[i];
  }
}

// Real token ids from the checkpoint's own vocab.json, so the prompts below
// mean something rather than being arbitrary integers.
constexpr int32_t kThe = 785;
constexpr int32_t kCapital = 6722;
constexpr int32_t kOf = 315;
constexpr int32_t kFrance = 9625;
constexpr int32_t kIs = 374;
constexpr int32_t kParis = 12095;
constexpr int32_t kBerlin = 19846;
constexpr int32_t kLondon = 7148;
constexpr int32_t kRome = 21718;

// The real test of M2, and the reason it is worth more than any statistical
// check on the logit distribution: "The capital of France is" must predict
// " Paris". Nothing about that survives a transposed projection, a mis-indexed
// RoPE, a leaking causal mask, a wrong norm epsilon or a swapped gate/up --
// each of those still yields finite, plausibly-shaped logits, and none of them
// gets the right token. It is a reference check that needs no reference.
TEST_F(Qwen2ForwardTest, PredictsParisAsTheCapitalOfFrance) {
  std::vector<float> logits;
  ASSERT_TRUE(
      model_->ForwardLastLogits({kThe, kCapital, kOf, kFrance, kIs}, &logits)
          .ok());

  const auto argmax =
      std::distance(logits.begin(), std::max_element(logits.begin(),
                                                     logits.end()));

  EXPECT_EQ(argmax, kParis)
      << "top token was id " << argmax << " (logit " << logits[argmax]
      << "); Paris scored " << logits[kParis];

  // And by a clear margin over other plausible capitals, which is what says the
  // model is reasoning about France rather than about capitals in general.
  for (const int32_t other : {kBerlin, kLondon, kRome}) {
    EXPECT_GT(logits[kParis], logits[other] + 1.0f)
        << "Paris=" << logits[kParis] << " vs id " << other << "="
        << logits[other];
  }
}

// A trained model's distribution is peaked; a stack whose signal has been
// destroyed goes flat. The threshold is the Gumbel expectation for the maximum
// of N standard normals -- sqrt(2 ln N) is about 4.9 for this vocabulary -- so
// anything at or below that is indistinguishable from noise.
TEST_F(Qwen2ForwardTest, LogitsAreMorePeakedThanNoise) {
  std::vector<float> logits;
  ASSERT_TRUE(
      model_->ForwardLastLogits({kThe, kCapital, kOf, kFrance, kIs}, &logits)
          .ok());

  const float max = *std::max_element(logits.begin(), logits.end());
  const double mean =
      std::accumulate(logits.begin(), logits.end(), 0.0) / logits.size();

  double var = 0.0;
  for (const float v : logits) var += (v - mean) * (v - mean);
  const double sd = std::sqrt(var / logits.size());

  const double noise_floor =
      std::sqrt(2.0 * std::log(static_cast<double>(logits.size())));

  EXPECT_GT((max - mean) / sd, noise_floor)
      << "max=" << max << " mean=" << mean << " sd=" << sd
      << " noise_floor=" << noise_floor;
}

// The last position must depend on the whole prompt. If the causal mask leaked
// or attention ignored history, changing an earlier token would leave the final
// logits untouched.
TEST_F(Qwen2ForwardTest, EarlierTokensChangeTheFinalLogits) {
  std::vector<float> a, b;
  ASSERT_TRUE(model_->ForwardLastLogits({kThe, kCapital, kOf, kFrance, kIs}, &a).ok());
  ASSERT_TRUE(model_->ForwardLastLogits({kThe, kCapital, kOf, 9707, kIs}, &b).ok());

  ASSERT_EQ(a.size(), b.size());

  double diff = 0.0;
  for (size_t i = 0; i < a.size(); ++i) diff += std::abs(a[i] - b[i]);
  diff /= a.size();

  EXPECT_GT(diff, 0.01)
      << "changing the first token barely moved the final logits (" << diff
      << "); attention may not be seeing history";
}

// The forward pass must be a pure function of its input. A stale scratch buffer
// or an uninitialised residual shows up here and almost nowhere else.
TEST_F(Qwen2ForwardTest, IsDeterministicAcrossCalls) {
  std::vector<float> first, second;
  ASSERT_TRUE(model_->ForwardLastLogits({kThe, kCapital, kOf, kFrance, kIs}, &first).ok());
  ASSERT_TRUE(model_->ForwardLastLogits({kThe, kCapital, kOf, kFrance, kIs}, &second).ok());

  ASSERT_EQ(first.size(), second.size());
  for (size_t i = 0; i < first.size(); ++i) {
    ASSERT_FLOAT_EQ(first[i], second[i]) << "at " << i;
  }
}

// Prefix invariance: with no cache, the logits at position i depend only on
// tokens 0..i. So running a prefix must reproduce that prefix's rows exactly.
// This is the sharpest causality check available -- it fails if any position
// can see beyond itself.
TEST_F(Qwen2ForwardTest, PrefixRowsAreInvariantToWhatFollows) {
  const std::vector<int32_t> full = {kThe, kCapital, kOf, kFrance, kIs, kParis};
  const std::vector<int32_t> prefix(full.begin(), full.begin() + 3);

  std::vector<float> full_logits, prefix_logits;
  ASSERT_TRUE(model_->Forward(full, &full_logits).ok());
  ASSERT_TRUE(model_->Forward(prefix, &prefix_logits).ok());

  const int64_t vocab = model_->config().vocab_size;
  ASSERT_EQ(prefix_logits.size(), prefix.size() * vocab);

  // Compare the last row of the prefix run against the same row of the full
  // run. Exactness is not expected -- attention sums over a different number of
  // keys in fp32 and the reduction order differs -- but they must agree closely.
  const size_t row = prefix.size() - 1;

  double worst = 0.0;
  for (int64_t i = 0; i < vocab; ++i) {
    worst = std::max<double>(
        worst, std::abs(full_logits[row * vocab + i] -
                        prefix_logits[row * vocab + i]));
  }

  EXPECT_LT(worst, 0.5)
      << "position " << row << " changed by " << worst
      << " when later tokens were appended; the causal mask is leaking";
}

TEST_F(Qwen2ForwardTest, RejectsOutOfVocabularyTokens) {
  std::vector<float> logits;

  EXPECT_EQ(model_->ForwardLastLogits({0, 999999999}, &logits).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(model_->ForwardLastLogits({}, &logits).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace
}  // namespace inferx::model
