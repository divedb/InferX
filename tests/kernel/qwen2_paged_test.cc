// Cached generation against full recompute.
//
// This is the statement that makes M3's cache trustworthy: prefilling a prompt
// and then decoding token by token must produce the same logits as re-running
// the whole prefix from scratch every time, which is what M2 did. If the two
// diverge, the cache is lying about history -- a rotated key stored at the
// wrong position, a block table off by one, a slot overwritten -- and every
// one of those produces fluent, wrong output that no smaller test catches.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/kv_cache.h"
#include "inferx/model/qwen2.h"

namespace inferx::model {
namespace {

constexpr int32_t kThe = 785;
constexpr int32_t kCapital = 6722;
constexpr int32_t kOf = 315;
constexpr int32_t kFrance = 9625;
constexpr int32_t kIs = 374;
constexpr int32_t kParis = 12095;

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

/// Minimal stand-in for what the scheduler will own: one sequence's blocks and
/// length, and the arithmetic that turns a position into a cache slot.
class TestSequence {
 public:
  TestSequence(KvBlockPool* pool, int64_t max_blocks)
      : pool_(pool), table_(pool->block_size()), max_blocks_(max_blocks) {}

  ~TestSequence() { (void)pool_->FreeBlocks(table_.blocks()); }

  /// Grows the block table so `total` tokens fit.
  Status Reserve(int64_t total) {
    while (table_.capacity_tokens() < total) {
      INFERX_ASSIGN_OR_RETURN(const int32_t block, pool_->AllocateBlock());
      table_.Append(block);
    }
    return OkStatus();
  }

  /// Builds the batch for `ids` starting at position `start`.
  StatusOr<ForwardBatch> MakeBatch(const std::vector<int32_t>& ids,
                                   int64_t start) {
    INFERX_RETURN_IF_ERROR(
        Reserve(start + static_cast<int64_t>(ids.size())));

    ForwardBatch batch;
    batch.token_ids = ids;
    batch.num_seqs = 1;
    batch.max_blocks_per_seq = max_blocks_;

    for (size_t i = 0; i < ids.size(); ++i) {
      const int64_t pos = start + static_cast<int64_t>(i);

      int32_t block = 0;
      int64_t slot = 0;
      if (!table_.Locate(pos, &block, &slot)) {
        return InternalError("position ", pos, " is not reserved");
      }

      batch.positions.push_back(static_cast<int32_t>(pos));
      batch.seq_of_token.push_back(0);
      batch.slots.push_back(
          static_cast<int32_t>(block * pool_->block_size() + slot));
    }

    batch.block_table.assign(static_cast<size_t>(max_blocks_), 0);
    for (size_t i = 0; i < table_.blocks().size(); ++i) {
      batch.block_table[i] = table_.blocks()[i];
    }

    // Only the final token's logits: that is what sampling needs.
    batch.logits_indices = {static_cast<int32_t>(ids.size() - 1)};

    return batch;
  }

 private:
  KvBlockPool* pool_;
  BlockTable table_;
  int64_t max_blocks_;
};

int64_t Argmax(const std::vector<float>& v) {
  return std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

class PagedModelTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!CudaAvailable()) return;

    auto loaded = Qwen2Model::LoadFromDirectory(CheckpointDir());
    if (!loaded.ok()) {
      status_ = loaded.status();
      return;
    }

    model_ = new Qwen2Model(*std::move(loaded));

    // 256 blocks x 16 tokens = 4096 tokens of cache across all 36 layers.
    status_ = model_->AttachKvCache(256, 16);
  }

  static void TearDownTestSuite() {
    delete model_;
    model_ = nullptr;
  }

  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";
    if (model_ == nullptr || !status_.ok()) {
      GTEST_SKIP() << "model unavailable: " << status_;
    }
  }

  static Qwen2Model* model_;
  static Status status_;
};

Qwen2Model* PagedModelTest::model_ = nullptr;
Status PagedModelTest::status_ = OkStatus();

TEST_F(PagedModelTest, PoolIsSizedAsRequested) {
  ASSERT_NE(model_->kv_pool(), nullptr);

  EXPECT_EQ(model_->kv_pool()->num_blocks(), 256);
  EXPECT_EQ(model_->kv_pool()->block_size(), 16);
  EXPECT_EQ(model_->kv_pool()->free_blocks(), 256);

  // 36 layers x 256 blocks x 2 entries x 16 tokens x 2 kv heads x 128 dims x 2 B
  EXPECT_EQ(model_->kv_pool()->bytes(), 36ull * 256 * 2 * 16 * 2 * 128 * 2);
}

// Prefill the whole prompt in one step, and the answer must be the one M2's
// full recompute gives.
TEST_F(PagedModelTest, PrefillMatchesFullRecompute) {
  const std::vector<int32_t> prompt = {kThe, kCapital, kOf, kFrance, kIs};

  std::vector<float> recomputed;
  ASSERT_TRUE(model_->ForwardLastLogits(prompt, &recomputed).ok());

  TestSequence seq(model_->kv_pool(), 8);
  auto batch = seq.MakeBatch(prompt, 0);
  ASSERT_TRUE(batch.ok()) << batch.status();

  std::vector<float> paged;
  ASSERT_TRUE(model_->Step(*batch, &paged).ok());

  ASSERT_EQ(paged.size(), recomputed.size());
  EXPECT_EQ(Argmax(paged), kParis);
  EXPECT_EQ(Argmax(paged), Argmax(recomputed));

  double worst = 0;
  for (size_t i = 0; i < paged.size(); ++i) {
    worst = std::max<double>(worst, std::abs(paged[i] - recomputed[i]));
  }

  // FlashInfer tiles and reduces attention differently from the deliberately
  // simple full-recompute kernel. Its kernel-level conformance test bounds one
  // layer to 0.00195 here; across 36 bf16 layers that becomes a few output
  // ulps. Keep this tight enough to catch a layout or masking error (those move
  // logits by order one), while separately requiring the exact sampled token
  // above.
  EXPECT_LT(worst, 0.25) << "worst logit difference " << worst;
}

// The real test: prefill a prefix, then feed tokens one at a time, and compare
// each step against a full recompute of everything seen so far. A cache that is
// subtly wrong drifts, so checking every step catches it at the step it starts
// rather than several tokens later.
TEST_F(PagedModelTest, IncrementalDecodeMatchesRecomputeAtEveryStep) {
  const std::vector<int32_t> full = {kThe, kCapital, kOf, kFrance, kIs, kParis};
  const int64_t prefill_len = 3;

  TestSequence seq(model_->kv_pool(), 8);

  const std::vector<int32_t> prefix(full.begin(), full.begin() + prefill_len);
  auto batch = seq.MakeBatch(prefix, 0);
  ASSERT_TRUE(batch.ok()) << batch.status();

  std::vector<float> paged;
  ASSERT_TRUE(model_->Step(*batch, &paged).ok());

  {
    std::vector<float> recomputed;
    ASSERT_TRUE(model_->ForwardLastLogits(prefix, &recomputed).ok());
    EXPECT_EQ(Argmax(paged), Argmax(recomputed)) << "after prefill";
  }

  for (int64_t next = prefill_len;
       next < static_cast<int64_t>(full.size()); ++next) {
    // One token, at its absolute position, attending over everything cached.
    auto step = seq.MakeBatch({full[static_cast<size_t>(next)]}, next);
    ASSERT_TRUE(step.ok()) << step.status();

    ASSERT_TRUE(model_->Step(*step, &paged).ok());

    const std::vector<int32_t> so_far(full.begin(), full.begin() + next + 1);
    std::vector<float> recomputed;
    ASSERT_TRUE(model_->ForwardLastLogits(so_far, &recomputed).ok());

    ASSERT_EQ(paged.size(), recomputed.size());

    EXPECT_EQ(Argmax(paged), Argmax(recomputed))
        << "decode step at position " << next << " disagrees with a recompute "
        << "of the same " << (next + 1) << " tokens";

    // The two paths are not bit-identical by construction, and it is worth
    // being precise about why: a recompute of n tokens runs its projections as
    // an [n, 2048] GEMM while a decode step runs [1, 2048], and cuBLASLt picks
    // a different algorithm for each -- different tiling, different reduction
    // order, different rounding. The cached keys themselves *are* identical:
    // they are written after RoPE and read back unmodified.
    //
    // So the bound is stated in the same units M2 established against
    // HuggingFace, where our bf16 stack sat 0.053-0.094 sd from fp32. Anything
    // inside that budget is arithmetic; anything outside it is a cache bug.
    double worst = 0;
    double sum = 0;
    for (size_t i = 0; i < paged.size(); ++i) {
      worst = std::max<double>(worst, std::abs(paged[i] - recomputed[i]));
      sum += recomputed[i];
    }

    const double mean = sum / recomputed.size();
    double var = 0;
    for (const float v : recomputed) var += (v - mean) * (v - mean);
    const double sd = std::sqrt(var / recomputed.size());

    EXPECT_LT(worst / sd, 0.1)
        << "position " << next << ": worst difference " << worst << " = "
        << (worst / sd) << " sd, beyond what bf16 rounding explains";
  }
}

// Generation: decode past the prompt and check the text is coherent. "The
// capital of France is" should continue with Paris and then keep going rather
// than degenerating, which is what a cache that slowly corrupts looks like.
TEST_F(PagedModelTest, GeneratesACoherentContinuation) {
  std::vector<int32_t> tokens = {kThe, kCapital, kOf, kFrance, kIs};

  TestSequence seq(model_->kv_pool(), 8);

  auto batch = seq.MakeBatch(tokens, 0);
  ASSERT_TRUE(batch.ok()) << batch.status();

  std::vector<float> logits;
  ASSERT_TRUE(model_->Step(*batch, &logits).ok());

  std::vector<int32_t> generated;

  for (int step = 0; step < 6; ++step) {
    const int32_t next = static_cast<int32_t>(Argmax(logits));
    generated.push_back(next);
    tokens.push_back(next);

    auto s = seq.MakeBatch({next},
                           static_cast<int64_t>(tokens.size()) - 1);
    ASSERT_TRUE(s.ok()) << s.status();
    ASSERT_TRUE(model_->Step(*s, &logits).ok());
  }

  EXPECT_EQ(generated[0], kParis) << "first generated token was "
                                  << generated[0];

  // Degenerate repetition is what a corrupted cache produces: the same token
  // forever, because history stopped mattering.
  const bool all_same = std::all_of(
      generated.begin(), generated.end(),
      [&](int32_t t) { return t == generated[0]; });

  EXPECT_FALSE(all_same) << "generation collapsed to a single repeated token";
}

// Two sequences interleaved in one pool must not see each other's keys. Their
// blocks are necessarily interleaved, which is exactly the arrangement where a
// block-table indexing bug shows up.
TEST_F(PagedModelTest, ConcurrentSequencesStayIsolated) {
  const std::vector<int32_t> a = {kThe, kCapital, kOf, kFrance, kIs};
  const std::vector<int32_t> b = {kThe, kCapital, kOf, kFrance, kIs, kParis};

  TestSequence sa(model_->kv_pool(), 8);
  TestSequence sb(model_->kv_pool(), 8);

  // Interleave the prefills so the two sequences' blocks alternate.
  auto ba = sa.MakeBatch(a, 0);
  auto bb = sb.MakeBatch(b, 0);
  ASSERT_TRUE(ba.ok() && bb.ok());

  std::vector<float> la, lb;
  ASSERT_TRUE(model_->Step(*ba, &la).ok());
  ASSERT_TRUE(model_->Step(*bb, &lb).ok());

  std::vector<float> ra, rb;
  ASSERT_TRUE(model_->ForwardLastLogits(a, &ra).ok());
  ASSERT_TRUE(model_->ForwardLastLogits(b, &rb).ok());

  EXPECT_EQ(Argmax(la), Argmax(ra)) << "sequence A was disturbed";
  EXPECT_EQ(Argmax(lb), Argmax(rb)) << "sequence B was disturbed";
  EXPECT_EQ(Argmax(la), kParis);
}

TEST_F(PagedModelTest, RejectsMalformedBatches) {
  ForwardBatch empty;
  std::vector<float> logits;
  EXPECT_EQ(model_->Step(empty, &logits).code(),
            absl::StatusCode::kInvalidArgument);

  ForwardBatch bad;
  bad.token_ids = {kThe};
  bad.positions = {0};
  bad.seq_of_token = {0};
  bad.slots = {999999};  // outside the pool
  bad.block_table = {0};
  bad.num_seqs = 1;
  bad.max_blocks_per_seq = 1;
  bad.logits_indices = {0};

  EXPECT_EQ(model_->Step(bad, &logits).code(),
            absl::StatusCode::kInvalidArgument);
}

// FP8 weights: does the quantized model still work, and by how much does it
// differ?
//
// This is deliberately not compared against the HuggingFace reference. FP8
// weights are a different numerical model -- one scale per tensor, three
// mantissa bits -- and holding them to a bf16 tolerance would be asserting that
// quantization does nothing, which is the opposite of what it does. What is
// asserted is that the model still answers correctly and that the divergence
// from bf16 is bounded and reported.
TEST_F(PagedModelTest, Fp8WeightsStillAnswerCorrectly) {
  const std::vector<int32_t> prompt = {kThe, kCapital, kOf, kFrance, kIs};

  // bf16 first, as the thing to compare against.
  std::vector<float> bf16_logits;
  {
    TestSequence seq(model_->kv_pool(), 8);
    auto batch = seq.MakeBatch(prompt, 0);
    ASSERT_TRUE(batch.ok()) << batch.status();
    ASSERT_TRUE(model_->Step(*batch, &bf16_logits).ok());
  }

  ASSERT_FALSE(model_->weights_are_f8());

  const size_t before = model_->WeightBytes();
  const Status quantized = model_->QuantizeWeightsToF8();
  ASSERT_TRUE(quantized.ok()) << quantized;
  ASSERT_TRUE(model_->weights_are_f8());

  std::vector<float> f8_logits;
  {
    TestSequence seq(model_->kv_pool(), 8);
    auto batch = seq.MakeBatch(prompt, 0);
    ASSERT_TRUE(batch.ok()) << batch.status();
    const Status stepped = model_->Step(*batch, &f8_logits);
    ASSERT_TRUE(stepped.ok()) << stepped;
  }

  ASSERT_EQ(f8_logits.size(), bf16_logits.size());

  // The answer must survive. This is the assertion that matters: a
  // quantization that breaks the model shows up here and nowhere subtler.
  EXPECT_EQ(Argmax(f8_logits), kParis)
      << "FP8 weights changed the answer; top token was "
      << Argmax(f8_logits);

  // And the divergence, reported rather than merely bounded, because the number
  // is what tells you whether FP8 is usable for a given application.
  double worst = 0, sum = 0;
  for (size_t i = 0; i < f8_logits.size(); ++i) {
    worst = std::max<double>(worst, std::abs(f8_logits[i] - bf16_logits[i]));
    sum += bf16_logits[i];
  }

  const double mean = sum / bf16_logits.size();
  double var = 0;
  for (const float v : bf16_logits) var += (v - mean) * (v - mean);
  const double sd = std::sqrt(var / bf16_logits.size());

  std::printf("  fp8 vs bf16: worst |diff| %.4f = %.3f sd; weights %.2f GB -> "
              "%.2f GB\n",
              worst, worst / sd, before / 1e9,
              model_->WeightBytes() / 1e9);

  // Loose on purpose, and stated as such: three mantissa bits over 36 layers
  // is a real perturbation, and the bound exists to catch a broken
  // quantization rather than to certify accuracy. The printed number is the
  // useful output.
  EXPECT_LT(worst / sd, 3.0) << "FP8 divergence is beyond what per-tensor "
                                "quantization should produce";
}

// The overlap pipeline, against the synchronous path it replaces.
//
// §5.2 depth 1: issue a step and return without waiting, so the host can build
// the next one while the GPU works. T6 keeps depth 0 as the correctness
// reference, and this is that comparison -- the two must generate identical
// tokens, because nothing about the arithmetic changed.
//
// What makes it possible is sampling on the device. The argmax writes each
// token straight into the buffer the next replay reads, so the host never has
// to learn the value to keep issuing work; it only needs positions and slots,
// which follow from the step count alone.
TEST_F(PagedModelTest, OverlappedGenerationMatchesSynchronous) {
  const std::vector<int32_t> prompt = {kThe, kCapital, kOf, kFrance, kIs};
  constexpr int kGenerate = 12;

  // --- depth 0: sample on the host, wait every step --------------------------
  std::vector<int32_t> synchronous;
  {
    TestSequence seq(model_->kv_pool(), 16);
    auto batch = seq.MakeBatch(prompt, 0);
    ASSERT_TRUE(batch.ok()) << batch.status();

    std::vector<float> logits;
    ASSERT_TRUE(model_->Step(*batch, &logits).ok());

    for (int i = 0; i < kGenerate; ++i) {
      const int32_t next = static_cast<int32_t>(Argmax(logits));
      synchronous.push_back(next);

      auto step = seq.MakeBatch(
          {next}, static_cast<int64_t>(prompt.size()) + i);
      ASSERT_TRUE(step.ok()) << step.status();
      ASSERT_TRUE(model_->Step(*step, &logits).ok());
    }
  }

  // --- depth 1: sample on the device, never wait inside the loop -------------
  ASSERT_TRUE(model_->EnableDeviceSampling(8).ok());

  std::vector<int32_t> overlapped;
  {
    TestSequence seq(model_->kv_pool(), 16);
    auto batch = seq.MakeBatch(prompt, 0);
    ASSERT_TRUE(batch.ok()) << batch.status();

    const Status first = model_->StepAsync(*batch);
    ASSERT_TRUE(first.ok()) << first;

    for (int i = 0; i < kGenerate; ++i) {
      // Built *before* the previous step's result is read: positions and slots
      // do not depend on which token came out, only on how many have. This is
      // the speculation §5.2 describes, and it is exact rather than a guess --
      // one token per sequence per step.
      auto step = seq.MakeBatch(
          {0}, static_cast<int64_t>(prompt.size()) + i);
      ASSERT_TRUE(step.ok()) << step.status();
      step->tokens_from_device = true;  // the sampler already wrote it

      std::vector<int32_t> tokens;
      ASSERT_TRUE(model_->AwaitStep(&tokens).ok());
      ASSERT_EQ(tokens.size(), 1u);
      overlapped.push_back(tokens[0]);

      // The token id in the batch is ignored for a device-sampled step: the
      // buffer already holds what the sampler wrote.
      ASSERT_TRUE(model_->StepAsync(*step).ok());
    }

    std::vector<int32_t> drain;
    ASSERT_TRUE(model_->AwaitStep(&drain).ok());
  }

  ASSERT_EQ(overlapped.size(), synchronous.size());
  EXPECT_EQ(overlapped, synchronous)
      << "the overlapped pipeline generated different tokens than the "
         "synchronous reference";
  EXPECT_EQ(overlapped[0], kParis);
}

}  // namespace
}  // namespace inferx::model
