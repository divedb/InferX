// The scheduler's whole lifecycle, on a machine with no GPU.
//
// That this file is in tests/unit rather than tests/kernel is the point. §3.1
// calls the scheduler/executor split the highest-leverage testability decision
// in the design, and this is what it buys: admission, block accounting,
// batching, stop conditions, cancellation and out-of-memory are all exercised
// against a host-allocated pool, with no device present and no forward pass
// involved. A bug in any of them is found here rather than inside a 6 GB model.

#include "inferx/scheduler/scheduler.h"

#include <algorithm>
#include <memory>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>

#include "absl/container/flat_hash_map.h"
#include "inferx/core/kv_cache.h"

namespace inferx::scheduler {
namespace {

using model::ForwardBatch;

/// A pool on the host. The free list is the same code the GPU path runs.
StatusOr<KvBlockPool> HostPool(int64_t blocks, int64_t block_size = 4) {
  KvLayout layout;
  layout.entries_per_token = 2;
  layout.kv_heads = 1;
  layout.head_dim = 8;
  layout.dtype = DataType::kBFloat16;

  return KvBlockPool::Create(/*num_layers=*/1, blocks, block_size, layout,
                             DeviceId::Cpu());
}

SamplingParams Params(int32_t max_tokens,
                      std::vector<int32_t> stops = {}) {
  SamplingParams p;
  p.max_tokens = max_tokens;
  p.stop_tokens = std::move(stops);
  return p;
}

/// Nothing was lost: every block is free, or held by the prefix cache, and
/// none is still claimed by a sequence.
///
/// The plain `free_blocks() == num_blocks()` this replaces stopped being true
/// once §6.3 landed, and rightly so -- a finished sequence's complete blocks go
/// into the radix tree rather than back to the free list. They are still
/// reclaimable, which is what `blocks_in_use()` reporting zero means.
void ExpectNothingLeaked(const KvBlockPool& pool, const Scheduler& sched) {
  EXPECT_EQ(sched.blocks_in_use(), 0) << "a finished sequence kept its blocks";
  EXPECT_EQ(pool.free_blocks() + sched.cached_blocks(), pool.num_blocks())
      << "blocks belong to neither a sequence, the cache, nor the free list";
}

class SchedulerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto p = HostPool(64);
    ASSERT_TRUE(p.ok()) << p.status();
    pool_ = std::make_unique<KvBlockPool>(*std::move(p));

    SchedulerConfig config;
    config.max_running = 4;
    config.max_batch_tokens = 64;
    config.max_seq_len = 64;

    auto s = Scheduler::Create(config, pool_.get());
    ASSERT_TRUE(s.ok()) << s.status();
    sched_ = std::make_unique<Scheduler>(*std::move(s));
  }

  std::unique_ptr<KvBlockPool> pool_;
  std::unique_ptr<Scheduler> sched_;
};

TEST_F(SchedulerTest, IdleSchedulerProducesAnEmptyBatch) {
  EXPECT_FALSE(sched_->HasWork());

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

  EXPECT_EQ(batch.num_tokens(), 0);
  EXPECT_TRUE(sched_->TakeCompleted().empty());
}

// The first step of a request is its prompt: every token, positions 0..n-1, and
// exactly one logits row -- the last. Asking for logits on prompt tokens would
// cost a [n, vocab] GEMM to discard nearly all of it.
TEST_F(SchedulerTest, FirstStepIsAPrefillOfTheWholePrompt) {
  ASSERT_TRUE(sched_->AddRequest(1, {10, 11, 12, 13, 14}, Params(4)).ok());
  EXPECT_EQ(sched_->num_waiting(), 1);

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

  EXPECT_EQ(sched_->num_running(), 1);
  EXPECT_EQ(sched_->num_waiting(), 0);

  EXPECT_EQ(batch.token_ids, (std::vector<int32_t>{10, 11, 12, 13, 14}));
  EXPECT_EQ(batch.positions, (std::vector<int32_t>{0, 1, 2, 3, 4}));
  EXPECT_EQ(batch.seq_of_token, (std::vector<int32_t>{0, 0, 0, 0, 0}));
  EXPECT_EQ(batch.logits_indices, (std::vector<int32_t>{4}));
  EXPECT_EQ(batch.num_seqs, 1);

  // 5 tokens at 4 per block is 2 blocks.
  EXPECT_EQ(sched_->blocks_in_use(), 2);

  ASSERT_TRUE(batch.Validate(1000, pool_->num_blocks() * pool_->block_size())
                  .ok());
}

// After the prefill, each step carries exactly one token per sequence, at the
// next position. This is the decode shape the whole engine is built around.
TEST_F(SchedulerTest, SubsequentStepsCarryOneTokenPerSequence) {
  ASSERT_TRUE(sched_->AddRequest(1, {10, 11, 12}, Params(3)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());
  ASSERT_EQ(batch.num_tokens(), 3);
  ASSERT_TRUE(sched_->CommitStep({99}).ok());

  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

  EXPECT_EQ(batch.token_ids, (std::vector<int32_t>{99}));
  EXPECT_EQ(batch.positions, (std::vector<int32_t>{3}));
  EXPECT_EQ(batch.logits_indices, (std::vector<int32_t>{0}));
}

// Slots must walk into the second block once the first fills, which is the
// arithmetic every cache write depends on.
TEST_F(SchedulerTest, SlotsCrossBlockBoundariesCorrectly) {
  ASSERT_TRUE(sched_->AddRequest(1, {1, 2, 3, 4, 5, 6}, Params(2)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

  ASSERT_EQ(batch.slots.size(), 6u);

  // Block size is 4. The first four tokens share a block; the next two are in
  // another, whichever the allocator handed back.
  const int32_t first_block = batch.slots[0] / 4;
  const int32_t second_block = batch.slots[4] / 4;

  EXPECT_NE(first_block, second_block);
  for (int i = 0; i < 4; ++i) EXPECT_EQ(batch.slots[i], first_block * 4 + i);
  EXPECT_EQ(batch.slots[4], second_block * 4 + 0);
  EXPECT_EQ(batch.slots[5], second_block * 4 + 1);
}

TEST_F(SchedulerTest, StopsOnAStopToken) {
  ASSERT_TRUE(sched_->AddRequest(7, {1, 2}, Params(100, {42})).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());
  ASSERT_TRUE(sched_->CommitStep({5}).ok());
  EXPECT_TRUE(sched_->TakeCompleted().empty());

  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());
  ASSERT_TRUE(sched_->CommitStep({42}).ok());

  const std::vector<Completion> done = sched_->TakeCompleted();
  ASSERT_EQ(done.size(), 1u);

  EXPECT_EQ(done[0].id, 7u);
  EXPECT_EQ(done[0].reason, FinishReason::kStopToken);
  // The stop token is part of the output; the caller decides whether to show it.
  EXPECT_EQ(done[0].output_tokens, (std::vector<int32_t>{5, 42}));

  EXPECT_FALSE(sched_->HasWork());
  EXPECT_EQ(sched_->blocks_in_use(), 0) << "finished sequence kept its blocks";
}

TEST_F(SchedulerTest, StopsAtMaxTokens) {
  ASSERT_TRUE(sched_->AddRequest(3, {1}, Params(3)).ok());

  ForwardBatch batch;
  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(sched_->PrepareStep(&batch).ok());
    ASSERT_EQ(batch.num_tokens(), 1) << "step " << i;
    ASSERT_TRUE(sched_->CommitStep({100 + i}).ok());
  }

  const std::vector<Completion> done = sched_->TakeCompleted();
  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].reason, FinishReason::kMaxTokens);
  EXPECT_EQ(done[0].output_tokens, (std::vector<int32_t>{100, 101, 102}));
}

// Blocks come back the moment a sequence finishes, not at some later sweep. A
// pool that holds dead blocks refuses admission to requests that could run.
TEST_F(SchedulerTest, BlocksAreReturnedAndReused) {
  const int64_t total = pool_->num_blocks();

  for (int r = 0; r < 3; ++r) {
    ASSERT_TRUE(sched_->AddRequest(static_cast<RequestId>(r), {1, 2, 3},
                                   Params(1))
                    .ok());

    ForwardBatch batch;
    ASSERT_TRUE(sched_->PrepareStep(&batch).ok());
    EXPECT_GT(sched_->blocks_in_use(), 0);

    ASSERT_TRUE(sched_->CommitStep({9}).ok());

    EXPECT_EQ(sched_->blocks_in_use(), 0) << "round " << r;
    EXPECT_EQ(pool_->free_blocks(), total) << "round " << r;
    sched_->TakeCompleted();
  }
}

TEST_F(SchedulerTest, RunsSeveralSequencesInOneBatch) {
  ASSERT_TRUE(sched_->AddRequest(1, {1, 2}, Params(5)).ok());
  ASSERT_TRUE(sched_->AddRequest(2, {3, 4, 5}, Params(5)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

  // Both prefills go in one batch: 2 + 3 tokens, two logits rows.
  EXPECT_EQ(sched_->num_running(), 2);
  EXPECT_EQ(batch.num_tokens(), 5);
  EXPECT_EQ(batch.num_seqs, 2);
  EXPECT_EQ(batch.logits_indices, (std::vector<int32_t>{1, 4}));
  EXPECT_EQ(batch.seq_of_token, (std::vector<int32_t>{0, 0, 1, 1, 1}));

  ASSERT_TRUE(sched_->CommitStep({50, 60}).ok());

  // Then both decode together, one token each.
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());
  EXPECT_EQ(batch.token_ids, (std::vector<int32_t>{50, 60}));
  EXPECT_EQ(batch.positions, (std::vector<int32_t>{2, 3}));
}

TEST_F(SchedulerTest, AdmissionIsCappedByMaxRunning) {
  for (int i = 0; i < 6; ++i) {
    ASSERT_TRUE(sched_->AddRequest(static_cast<RequestId>(i), {1, 2},
                                   Params(2))
                    .ok());
  }

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

  EXPECT_EQ(sched_->num_running(), 4);  // max_running
  EXPECT_EQ(sched_->num_waiting(), 2);
}

// FCFS (§8.3): the queue is a queue. Deliberately not shortest-job-first, which
// would reorder and destroy prefix-cache locality once M5 arrives.
TEST_F(SchedulerTest, AdmissionIsFirstComeFirstServed) {
  SchedulerConfig config;
  config.max_running = 1;
  config.max_batch_tokens = 64;
  config.max_seq_len = 64;

  auto s = Scheduler::Create(config, pool_.get());
  ASSERT_TRUE(s.ok());
  Scheduler sched = *std::move(s);

  ASSERT_TRUE(sched.AddRequest(11, {1, 2, 3, 4, 5, 6, 7, 8}, Params(1)).ok());
  ASSERT_TRUE(sched.AddRequest(22, {1}, Params(1)).ok());  // much shorter

  ForwardBatch batch;
  ASSERT_TRUE(sched.PrepareStep(&batch).ok());
  ASSERT_TRUE(sched.CommitStep({9}).ok());

  const std::vector<Completion> done = sched.TakeCompleted();
  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].id, 11u) << "the shorter request jumped the queue";
}

TEST_F(SchedulerTest, CancellingAQueuedRequestRetiresItUnrun) {
  ASSERT_TRUE(sched_->AddRequest(5, {1, 2, 3}, Params(4)).ok());
  sched_->Cancel(5);

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

  EXPECT_EQ(batch.num_tokens(), 0);

  const std::vector<Completion> done = sched_->TakeCompleted();
  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].reason, FinishReason::kCancelled);
  EXPECT_TRUE(done[0].output_tokens.empty());
  EXPECT_EQ(sched_->blocks_in_use(), 0);
}

TEST_F(SchedulerTest, CancellingARunningRequestFreesItsBlocks) {
  ASSERT_TRUE(sched_->AddRequest(6, {1, 2, 3, 4, 5}, Params(10)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());
  ASSERT_TRUE(sched_->CommitStep({7}).ok());
  ASSERT_GT(sched_->blocks_in_use(), 0);

  sched_->Cancel(6);
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

  EXPECT_EQ(sched_->blocks_in_use(), 0);
  EXPECT_FALSE(sched_->HasWork());

  const std::vector<Completion> done = sched_->TakeCompleted();
  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].reason, FinishReason::kCancelled);
  // Partial output survives cancellation -- the caller may already have streamed
  // it (§4, step 10).
  EXPECT_EQ(done[0].output_tokens, (std::vector<int32_t>{7}));
}

// Cancelling a request the scheduler has never heard of, or has already
// finished, must be silent: a disconnect races with everything.
TEST_F(SchedulerTest, CancellingAnUnknownRequestIsHarmless) {
  sched_->Cancel(999);

  ForwardBatch batch;
  EXPECT_TRUE(sched_->PrepareStep(&batch).ok());
  EXPECT_TRUE(sched_->TakeCompleted().empty());
}

// A prompt longer than the pool can ever hold is refused rather than queued
// forever, and the reason distinguishes it from a transient shortage.
TEST_F(SchedulerTest, PromptsBeyondMaxSeqLenAreRejected) {
  SchedulerConfig config;
  config.max_running = 2;
  config.max_batch_tokens = 1024;
  config.max_seq_len = 8;

  auto s = Scheduler::Create(config, pool_.get());
  ASSERT_TRUE(s.ok());
  Scheduler sched = *std::move(s);

  std::vector<int32_t> huge(20, 1);
  ASSERT_TRUE(sched.AddRequest(1, std::move(huge), Params(1)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched.PrepareStep(&batch).ok());

  const std::vector<Completion> done = sched.TakeCompleted();
  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].reason, FinishReason::kOutOfMemory);
  EXPECT_EQ(sched.blocks_in_use(), 0);
}

// A transient shortage leaves the request queued rather than failing it, and
// admission resumes when a running sequence frees its blocks. Without the
// rollback in Admit(), the partially-reserved blocks would leak.
TEST_F(SchedulerTest, ExhaustionDefersAdmissionRatherThanLeaking) {
  auto small = HostPool(/*blocks=*/2, /*block_size=*/4);
  ASSERT_TRUE(small.ok());
  KvBlockPool pool = *std::move(small);

  SchedulerConfig config;
  config.max_running = 4;
  config.max_batch_tokens = 64;
  config.max_seq_len = 64;

  auto s = Scheduler::Create(config, &pool);
  ASSERT_TRUE(s.ok());
  Scheduler sched = *std::move(s);

  // 8 tokens fills both blocks; the second request cannot fit.
  ASSERT_TRUE(sched.AddRequest(1, std::vector<int32_t>(8, 1), Params(1)).ok());
  ASSERT_TRUE(sched.AddRequest(2, std::vector<int32_t>(4, 2), Params(1)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched.PrepareStep(&batch).ok());

  EXPECT_EQ(sched.num_running(), 1);
  EXPECT_EQ(sched.num_waiting(), 1) << "the second request should still queue";
  EXPECT_EQ(pool.free_blocks(), 0);

  // Finish the first; its blocks come back and the second is admitted. They
  // come back into the prefix cache rather than onto the free list (§6.3),
  // which is why this asks whether a sequence still holds them rather than
  // whether they are free -- admission evicts to get at them.
  ASSERT_TRUE(sched.CommitStep({9}).ok());
  EXPECT_EQ(sched.blocks_in_use(), 0)
      << "blocks leaked on the deferred admission";

  ASSERT_TRUE(sched.PrepareStep(&batch).ok());
  EXPECT_EQ(sched.num_running(), 1);
  EXPECT_EQ(batch.token_ids, (std::vector<int32_t>{2, 2, 2, 2}));
}

// ---------------------------------------------------------------------------
// Chunked prefill (§8.1).
// ---------------------------------------------------------------------------

/// A scheduler with a deliberately tiny token budget, so a handful of tokens
/// is enough to exercise chunking.
std::unique_ptr<Scheduler> ChunkingScheduler(KvBlockPool* pool,
                                             int64_t max_batch_tokens,
                                             int64_t max_running = 4) {
  SchedulerConfig config;
  config.max_running = max_running;
  config.max_batch_tokens = max_batch_tokens;
  config.max_seq_len = 64;

  auto s = Scheduler::Create(config, pool);
  if (!s.ok()) return nullptr;
  return std::make_unique<Scheduler>(*std::move(s));
}

// The budget is spent rather than left on the table: what does not fit whole is
// cut to the remainder. This is the test that used to assert the opposite --
// that a prompt too big for what was left waited for a step of its own -- and
// inverting it is the point of chunking.
TEST_F(SchedulerTest, APromptTakesWhatIsLeftOfTheBudgetRatherThanDeferring) {
  auto sched = ChunkingScheduler(pool_.get(), /*max_batch_tokens=*/4);
  ASSERT_NE(sched, nullptr);

  ASSERT_TRUE(sched->AddRequest(1, {1, 2, 3}, Params(2)).ok());
  ASSERT_TRUE(sched->AddRequest(2, {4, 5, 6}, Params(2)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched->PrepareStep(&batch).ok());

  // Request 1 takes 3 of 4; request 2 takes the remaining 1 as a chunk.
  EXPECT_EQ(batch.token_ids, (std::vector<int32_t>{1, 2, 3, 4}));
  EXPECT_EQ(batch.seq_of_token, (std::vector<int32_t>{0, 0, 0, 1}));
  EXPECT_EQ(batch.positions, (std::vector<int32_t>{0, 1, 2, 0}));

  // Only request 1 finished its prompt, so only it has a logits row.
  EXPECT_EQ(batch.logits_indices, (std::vector<int32_t>{2}));

  // Both lengths are stated, and request 2's says 1 -- it has one key in the
  // cache and two prompt tokens still to come.
  EXPECT_EQ(batch.seq_lens, (std::vector<int32_t>{3, 1}));
}

// A prompt longer than the whole per-step budget was unservable before
// chunking: it could never fit a step, so it sat in the queue forever. Now it
// runs in pieces, and the pieces cover it exactly once.
TEST_F(SchedulerTest, APromptLongerThanTheBudgetIsServedInChunks) {
  auto sched = ChunkingScheduler(pool_.get(), /*max_batch_tokens=*/4);
  ASSERT_NE(sched, nullptr);

  std::vector<int32_t> prompt(15);
  for (size_t i = 0; i < prompt.size(); ++i) {
    prompt[i] = static_cast<int32_t>(100 + i);
  }

  ASSERT_TRUE(sched->AddRequest(1, prompt, Params(1)).ok());

  ForwardBatch batch;
  std::vector<int32_t> seen;
  std::vector<int32_t> seen_positions;
  int chunks = 0;

  // Run until the prompt produces its first token.
  while (true) {
    ASSERT_TRUE(sched->PrepareStep(&batch).ok());
    ASSERT_FALSE(batch.token_ids.empty());
    ++chunks;
    ASSERT_LT(chunks, 20) << "prefill did not converge";

    seen.insert(seen.end(), batch.token_ids.begin(), batch.token_ids.end());
    seen_positions.insert(seen_positions.end(), batch.positions.begin(),
                          batch.positions.end());

    EXPECT_LE(batch.num_tokens(), 4) << "chunk overran the budget";

    if (!batch.logits_indices.empty()) break;

    // A middle chunk asks for nothing and takes nothing back.
    ASSERT_TRUE(sched->CommitStep({}).ok());
  }

  // 15 tokens at 4 per step is four chunks, the last of them short.
  EXPECT_EQ(chunks, 4);
  EXPECT_EQ(seen, prompt) << "the chunks did not reconstruct the prompt";

  // Every position exactly once, in order: no gap, no token run twice.
  std::vector<int32_t> expected_positions(prompt.size());
  for (size_t i = 0; i < expected_positions.size(); ++i) {
    expected_positions[i] = static_cast<int32_t>(i);
  }
  EXPECT_EQ(seen_positions, expected_positions);

  // The last chunk is the one that samples, and it samples once.
  EXPECT_EQ(batch.logits_indices.size(), 1u);
  EXPECT_EQ(batch.logits_indices[0], batch.num_tokens() - 1);
  EXPECT_EQ(batch.seq_lens, (std::vector<int32_t>{15}));
}

// §8.1's priority rule. A decoding sequence must not wait behind someone
// else's prompt, because bounding that wait is the reason chunking exists.
TEST_F(SchedulerTest, DecodesAreServedBeforePrefillChunks) {
  auto sched = ChunkingScheduler(pool_.get(), /*max_batch_tokens=*/4);
  ASSERT_NE(sched, nullptr);

  // Request 1 gets running and starts generating.
  ASSERT_TRUE(sched->AddRequest(1, {1, 2}, Params(10)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched->PrepareStep(&batch).ok());
  ASSERT_EQ(batch.num_tokens(), 2);
  ASSERT_TRUE(sched->CommitStep({50}).ok());

  // Now a long prompt arrives. It is far bigger than the budget, so without
  // decode-first it would take the whole step and stall request 1.
  ASSERT_TRUE(sched->AddRequest(2, {7, 7, 7, 7, 7, 7, 7, 7}, Params(1)).ok());

  ASSERT_TRUE(sched->PrepareStep(&batch).ok());

  // Request 1's decode is present, and it is first: rows are grouped by
  // sequence in ascending sequence order, which the attention plan requires.
  EXPECT_EQ(batch.seq_of_token, (std::vector<int32_t>{0, 1, 1, 1}));
  EXPECT_EQ(batch.token_ids, (std::vector<int32_t>{50, 7, 7, 7}));

  // The decode took 1 of the 4 tokens; the prompt was chunked into the other 3
  // rather than being given the whole budget.
  EXPECT_EQ(batch.logits_indices, (std::vector<int32_t>{0}));
  EXPECT_EQ(batch.seq_lens, (std::vector<int32_t>{3, 3}));
}

// The equivalence that makes chunking safe: splitting a prompt changes how many
// steps it takes and nothing else. Same tokens, same positions, same slots, so
// the same keys land in the same places.
TEST_F(SchedulerTest, ChunkingChangesTheStepsAndNotTheWork) {
  const std::vector<int32_t> prompt = {5, 6, 7, 8, 9, 10, 11, 12, 13};

  // Collects (token, position, slot) for every prompt token, across however
  // many steps the budget makes it take.
  const auto run = [&](int64_t budget) {
    auto pool = HostPool(64);
    EXPECT_TRUE(pool.ok());
    auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

    auto sched = ChunkingScheduler(owned.get(), budget);
    EXPECT_NE(sched, nullptr);
    EXPECT_TRUE(sched->AddRequest(1, prompt, Params(1)).ok());

    std::vector<std::tuple<int32_t, int32_t, int32_t>> work;
    ForwardBatch batch;

    for (int step = 0; step < 20; ++step) {
      EXPECT_TRUE(sched->PrepareStep(&batch).ok());
      if (batch.token_ids.empty()) break;

      for (int64_t i = 0; i < batch.num_tokens(); ++i) {
        work.emplace_back(batch.token_ids[i], batch.positions[i],
                          batch.slots[i]);
      }

      if (!batch.logits_indices.empty()) break;
      EXPECT_TRUE(sched->CommitStep({}).ok());
    }

    return work;
  };

  const auto whole = run(/*budget=*/64);
  const auto chunked = run(/*budget=*/2);

  ASSERT_EQ(whole.size(), prompt.size());
  EXPECT_EQ(chunked, whole);
}

// A chunk in the middle of a prompt produces no logits, so it takes no sampled
// tokens back -- and offering it one is a caller bug worth catching.
TEST_F(SchedulerTest, CommitRejectsATokenForAChunkThatAskedForNone) {
  auto sched = ChunkingScheduler(pool_.get(), /*max_batch_tokens=*/2);
  ASSERT_NE(sched, nullptr);

  ASSERT_TRUE(sched->AddRequest(1, {1, 2, 3, 4, 5}, Params(2)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched->PrepareStep(&batch).ok());
  ASSERT_TRUE(batch.logits_indices.empty()) << "expected a middle chunk";

  EXPECT_EQ(sched->CommitStep({42}).code(), absl::StatusCode::kInvalidArgument);
}

// A chunked prompt still finishes, still generates the right number of tokens,
// and still gives its blocks back. The whole lifecycle, with the budget set
// low enough that every request is chunked.
TEST_F(SchedulerTest, ChunkedPromptsDrainToCompletion) {
  auto sched = ChunkingScheduler(pool_.get(), /*max_batch_tokens=*/3);
  ASSERT_NE(sched, nullptr);

  ASSERT_TRUE(sched->AddRequest(1, {1, 2, 3, 4, 5, 6, 7}, Params(4)).ok());
  ASSERT_TRUE(sched->AddRequest(2, {8, 9, 10, 11, 12}, Params(3)).ok());

  std::vector<Completion> all;
  int32_t next_token = 70;
  int steps = 0;

  while (sched->HasWork()) {
    ASSERT_LT(++steps, 200) << "step loop did not terminate";

    ForwardBatch batch;
    ASSERT_TRUE(sched->PrepareStep(&batch).ok());

    if (batch.num_tokens() == 0) {
      for (Completion& c : sched->TakeCompleted()) all.push_back(std::move(c));
      continue;
    }

    ASSERT_LE(batch.num_tokens(), 3) << "step " << steps << " overran budget";
    ASSERT_TRUE(batch.Validate(1000, pool_->num_blocks() * pool_->block_size())
                    .ok())
        << "step " << steps << ": "
        << batch.Validate(1000, pool_->num_blocks() * pool_->block_size());

    std::vector<int32_t> sampled(batch.logits_indices.size());
    for (int32_t& t : sampled) t = next_token++;

    ASSERT_TRUE(sched->CommitStep(sampled).ok());

    for (Completion& c : sched->TakeCompleted()) all.push_back(std::move(c));
  }

  for (Completion& c : sched->TakeCompleted()) all.push_back(std::move(c));

  ASSERT_EQ(all.size(), 2u);
  for (const Completion& c : all) {
    EXPECT_EQ(c.reason, FinishReason::kMaxTokens) << "request " << c.id;
    EXPECT_EQ(c.output_tokens.size(), c.id == 1 ? 4u : 3u)
        << "request " << c.id << " generated the wrong number of tokens";
  }

  ExpectNothingLeaked(*pool_, *sched);
}

// Every sequence in the batch has a declared length, including one the budget
// left out entirely. That sequence has no query row, so its length cannot be
// recovered from the batch any other way -- which is why ForwardBatch carries
// it rather than the model deriving it.
TEST_F(SchedulerTest, DeferredSequencesStillDeclareTheirLength) {
  // Budget 1, so a single decode consumes it and nothing else gets a token.
  auto sched = ChunkingScheduler(pool_.get(), /*max_batch_tokens=*/1);
  ASSERT_NE(sched, nullptr);

  ASSERT_TRUE(sched->AddRequest(1, {1}, Params(10)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched->PrepareStep(&batch).ok());
  ASSERT_EQ(batch.num_tokens(), 1);
  ASSERT_TRUE(sched->CommitStep({50}).ok());

  // Request 2 is admitted but the decode takes the only token of budget.
  ASSERT_TRUE(sched->AddRequest(2, {9, 9}, Params(1)).ok());

  ASSERT_TRUE(sched->PrepareStep(&batch).ok());

  ASSERT_EQ(batch.num_seqs, 2);
  EXPECT_EQ(batch.seq_of_token, (std::vector<int32_t>{0}))
      << "the decode should have taken the budget";

  // Request 1 has two keys cached; request 2 has none yet, and says so.
  EXPECT_EQ(batch.seq_lens, (std::vector<int32_t>{2, 0}));
}

// ---------------------------------------------------------------------------
// Preemption (§8.2).
// ---------------------------------------------------------------------------

/// A scheduler over a pool small enough that sequences have to compete for it.
///
/// The block size is 4, so every four generated tokens costs a block and the
/// pool can be drained in a handful of steps rather than thousands.
std::unique_ptr<Scheduler> TightScheduler(KvBlockPool* pool,
                                          int64_t max_running = 4) {
  SchedulerConfig config;
  config.max_running = max_running;
  config.max_batch_tokens = 64;
  config.max_seq_len = 64;

  auto s = Scheduler::Create(config, pool);
  if (!s.ok()) return nullptr;
  return std::make_unique<Scheduler>(*std::move(s));
}

/// Runs to completion with a deterministic fake executor, so that what a
/// sequence generates depends only on its own history and not on how the batch
/// happened to be composed. That is what makes "preempted output == unpreempted
/// output" a meaningful comparison.
///
/// The token is a hash of the request id and how many tokens it has produced,
/// which is exactly what a real model's greedy argmax is: a function of the
/// sequence's own contents.
std::vector<Completion> DrainDeterministically(Scheduler* sched,
                                               int max_steps = 4000) {
  std::vector<Completion> all;
  absl::flat_hash_map<RequestId, int> produced;

  for (int step = 0; step < max_steps && sched->HasWork(); ++step) {
    ForwardBatch batch;
    if (!sched->PrepareStep(&batch).ok()) break;

    if (batch.num_tokens() > 0) {
      std::vector<int32_t> sampled;
      sampled.reserve(batch.logits_indices.size());

      // The batch does not say which request each logits row belongs to, so
      // the deltas from the previous commit cannot be used here. Recover it the
      // way the model would: the row's sequence index into the block table.
      for (const int32_t row : batch.logits_indices) {
        const int32_t seq = batch.seq_of_token[static_cast<size_t>(row)];
        // Position is a stand-in for "the sequence's own contents".
        const int32_t pos = batch.positions[static_cast<size_t>(row)];
        sampled.push_back(100 + (pos * 7) % 50);
      }

      if (!sched->CommitStep(sampled).ok()) break;
    }

    for (Completion& c : sched->TakeCompleted()) all.push_back(std::move(c));
  }

  for (Completion& c : sched->TakeCompleted()) all.push_back(std::move(c));
  return all;
}

// The property that makes recompute preemption safe: a sequence that was
// evicted mid-generation and re-ran its own history produces exactly what it
// would have produced had it never been touched.
//
// This is the whole correctness claim of §8.2. A preempted sequence keeps its
// generated tokens and loses only its KV, so resuming means prefilling prompt
// *plus* what it already said and predicting from there. Get the resumption
// point wrong by one and it either repeats a token or skips one -- and either
// way it still returns a plausible-looking completion.
TEST_F(SchedulerTest, APreemptedSequenceProducesWhatItWouldHaveAnyway) {
  const std::vector<std::vector<int32_t>> prompts = {
      {1, 2, 3}, {4, 5}, {6, 7, 8, 9}, {10}};

  // Roomy: nothing competes, nothing is preempted.
  std::vector<Completion> unpreempted;
  {
    auto pool = HostPool(/*blocks=*/256);
    ASSERT_TRUE(pool.ok());
    auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

    auto sched = TightScheduler(owned.get());
    ASSERT_NE(sched, nullptr);

    for (size_t i = 0; i < prompts.size(); ++i) {
      ASSERT_TRUE(sched->AddRequest(static_cast<RequestId>(i + 1), prompts[i],
                                    Params(20))
                      .ok());
    }

    unpreempted = DrainDeterministically(sched.get());
    ASSERT_EQ(sched->preemptions(), 0) << "the control was itself preempted";
  }

  // Cramped: the same work through a pool that cannot hold it all at once.
  std::vector<Completion> preempted;
  int64_t preemptions = 0;
  {
    auto pool = HostPool(/*blocks=*/12);
    ASSERT_TRUE(pool.ok());
    auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

    auto sched = TightScheduler(owned.get());
    ASSERT_NE(sched, nullptr);

    for (size_t i = 0; i < prompts.size(); ++i) {
      ASSERT_TRUE(sched->AddRequest(static_cast<RequestId>(i + 1), prompts[i],
                                    Params(20))
                      .ok());
    }

    preempted = DrainDeterministically(sched.get());
    preemptions = sched->preemptions();

    ExpectNothingLeaked(*owned, *sched);
  }

  ASSERT_GT(preemptions, 0)
      << "the pool was not tight enough to force a preemption";

  // Sort both by id: preemption reorders completions, which is allowed.
  const auto by_id = [](const Completion& a, const Completion& b) {
    return a.id < b.id;
  };
  std::sort(unpreempted.begin(), unpreempted.end(), by_id);
  std::sort(preempted.begin(), preempted.end(), by_id);

  ASSERT_EQ(unpreempted.size(), prompts.size());
  ASSERT_EQ(preempted.size(), unpreempted.size());

  for (size_t i = 0; i < preempted.size(); ++i) {
    EXPECT_EQ(preempted[i].id, unpreempted[i].id);
    EXPECT_EQ(preempted[i].reason, unpreempted[i].reason)
        << "request " << preempted[i].id;
    EXPECT_EQ(preempted[i].output_tokens, unpreempted[i].output_tokens)
        << "request " << preempted[i].id << " came back different after "
        << preemptions << " preemptions";
  }
}

// A shortage costs some sequence its KV, not the whole batch its life. Before
// preemption this path returned an error out of PrepareStep, and the engine's
// only answer to that is to fail every request in flight.
TEST_F(SchedulerTest, ExhaustionPreemptsInsteadOfFailingTheStep) {
  auto pool = HostPool(/*blocks=*/8);
  ASSERT_TRUE(pool.ok());
  auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

  auto sched = TightScheduler(owned.get(), /*max_running=*/4);
  ASSERT_NE(sched, nullptr);

  // Four sequences generating long enough to exhaust 8 blocks between them.
  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(
        sched->AddRequest(static_cast<RequestId>(i + 1), {1, 2}, Params(30))
            .ok());
  }

  ForwardBatch batch;

  for (int step = 0; step < 500 && sched->HasWork(); ++step) {
    // Every step succeeds. That is the assertion.
    ASSERT_TRUE(sched->PrepareStep(&batch).ok()) << "step " << step;

    if (batch.num_tokens() == 0) continue;

    std::vector<int32_t> sampled(batch.logits_indices.size(), 55);
    ASSERT_TRUE(sched->CommitStep(sampled).ok()) << "step " << step;
    (void)sched->TakeCompleted();
  }

  EXPECT_GT(sched->preemptions(), 0) << "nothing was ever preempted";
  ExpectNothingLeaked(*owned, *sched);
}

// A preempted sequence goes to the *head* of the queue, not the back. It was
// admitted before everything still waiting and a client has already seen tokens
// from it, so letting newer requests overtake it is the one unfairness FCFS
// exists to prevent.
TEST_F(SchedulerTest, APreemptedSequenceGoesToTheFrontOfTheQueue) {
  auto pool = HostPool(/*blocks=*/6);
  ASSERT_TRUE(pool.ok());
  auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

  auto sched = TightScheduler(owned.get(), /*max_running=*/2);
  ASSERT_NE(sched, nullptr);

  ASSERT_TRUE(sched->AddRequest(1, {1, 2, 3}, Params(30)).ok());
  ASSERT_TRUE(sched->AddRequest(2, {4, 5, 6}, Params(30)).ok());

  ForwardBatch batch;

  // Run until something is preempted.
  int step = 0;
  for (; step < 500 && sched->preemptions() == 0; ++step) {
    ASSERT_TRUE(sched->PrepareStep(&batch).ok());
    if (batch.num_tokens() == 0) break;

    std::vector<int32_t> sampled(batch.logits_indices.size(), 55);
    ASSERT_TRUE(sched->CommitStep(sampled).ok());
  }

  ASSERT_GT(sched->preemptions(), 0) << "nothing was preempted in " << step
                                     << " steps";

  // A newcomer arrives while the preempted sequence is queued.
  ASSERT_TRUE(sched->AddRequest(99, {7}, Params(2)).ok());
  EXPECT_EQ(sched->num_waiting(), 2);

  // The preempted one is re-admitted first, so the newcomer never overtakes it.
  // Drive until the queue drains and check nothing starved.
  const std::vector<Completion> done = DrainDeterministically(sched.get());

  bool saw_99 = false;
  for (const Completion& c : done) {
    if (c.id == 99) saw_99 = true;
    EXPECT_NE(c.reason, FinishReason::kNotFinished) << "request " << c.id;
  }

  EXPECT_TRUE(saw_99) << "the newcomer never ran";
  ExpectNothingLeaked(*owned, *sched);
}

// Filling max_seq_len is not a shortage and preemption cannot help: the limit
// is on where a sequence's tokens may live, not on whether room exists. It
// retires with its own reason, and everyone else keeps running.
//
// This used to be the same ResourceExhausted as an empty pool, which is how one
// long-running sequence could take the whole server down with it.
TEST_F(SchedulerTest, FillingTheContextRetiresOnlyThatSequence) {
  auto pool = HostPool(/*blocks=*/256);
  ASSERT_TRUE(pool.ok());
  auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

  SchedulerConfig config;
  config.max_running = 4;
  config.max_batch_tokens = 64;
  config.max_seq_len = 16;  // 4 blocks of 4

  auto created = Scheduler::Create(config, owned.get());
  ASSERT_TRUE(created.ok());
  Scheduler sched = *std::move(created);

  // Asks for far more tokens than the context can hold.
  ASSERT_TRUE(sched.AddRequest(1, {1, 2}, Params(100)).ok());
  // And a modest one alongside it, which must be unaffected.
  ASSERT_TRUE(sched.AddRequest(2, {3, 4}, Params(3)).ok());

  ForwardBatch batch;
  std::vector<Completion> done;

  for (int step = 0; step < 200 && sched.HasWork(); ++step) {
    ASSERT_TRUE(sched.PrepareStep(&batch).ok()) << "step " << step;
    if (batch.num_tokens() == 0) continue;

    std::vector<int32_t> sampled(batch.logits_indices.size(), 55);
    ASSERT_TRUE(sched.CommitStep(sampled).ok());

    for (Completion& c : sched.TakeCompleted()) done.push_back(std::move(c));
  }

  for (Completion& c : sched.TakeCompleted()) done.push_back(std::move(c));

  ASSERT_EQ(done.size(), 2u);

  for (const Completion& c : done) {
    if (c.id == 1) {
      EXPECT_EQ(c.reason, FinishReason::kContextLimit);

      // One more than the context holds, and that is right rather than an
      // off-by-one. A 16-token context and a 2-token prompt leave room for 14
      // generated tokens *with KV*; the 15th is sampled from the 14th's logits
      // and retires the sequence before it is ever fed back, so it needs no
      // block. The token a client sees last is always one the cache never held.
      EXPECT_EQ(c.output_tokens.size(), 15u);
    } else {
      EXPECT_EQ(c.reason, FinishReason::kMaxTokens)
          << "the neighbouring request was collateral damage";
      EXPECT_EQ(c.output_tokens.size(), 3u);
    }
  }

  // No preemption happened: a full context is not a shortage.
  EXPECT_EQ(sched.preemptions(), 0);
  ExpectNothingLeaked(*owned, sched);
}

// The one case with nothing to preempt: a single sequence that has drained the
// pool by itself. There is no victim, so it retires rather than spinning.
TEST_F(SchedulerTest, TheLastSequenceStandingRetiresRatherThanSpinning) {
  auto pool = HostPool(/*blocks=*/3);
  ASSERT_TRUE(pool.ok());
  auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

  auto sched = TightScheduler(owned.get(), /*max_running=*/1);
  ASSERT_NE(sched, nullptr);

  ASSERT_TRUE(sched->AddRequest(1, {1, 2}, Params(60)).ok());

  const std::vector<Completion> done = DrainDeterministically(sched.get());

  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].reason, FinishReason::kOutOfMemory);

  // 3 blocks of 4 is 12 cached tokens, less the 2-token prompt, plus the final
  // sampled token that retires the sequence before it needs a slot.
  EXPECT_EQ(done[0].output_tokens.size(), 11u);

  EXPECT_EQ(sched->preemptions(), 0) << "there was nobody to preempt";
  ExpectNothingLeaked(*owned, *sched);
}

// ---------------------------------------------------------------------------
// Prefix caching (§6.3), where the scheduler meets the tree.
// ---------------------------------------------------------------------------

/// Drives a workload and reports both what came out and how much forward work
/// it took. Tokens forwarded is the number prefix caching is supposed to move.
struct Workload {
  std::vector<Completion> completions;
  int64_t tokens_forwarded = 0;
  int64_t steps = 0;
};

Workload RunWorkload(Scheduler* sched, int max_steps = 2000) {
  Workload out;

  for (int step = 0; step < max_steps && sched->HasWork(); ++step) {
    ForwardBatch batch;
    if (!sched->PrepareStep(&batch).ok()) break;

    if (batch.num_tokens() > 0) {
      ++out.steps;
      out.tokens_forwarded += batch.num_tokens();

      std::vector<int32_t> sampled;
      for (const int32_t row : batch.logits_indices) {
        // A function of the sequence's own position, so the "same input gives
        // the same output" comparison below means something.
        sampled.push_back(200 + (batch.positions[static_cast<size_t>(row)] * 3) % 40);
      }

      if (!sched->CommitStep(sampled).ok()) break;
    }

    for (Completion& c : sched->TakeCompleted()) {
      out.completions.push_back(std::move(c));
    }
  }

  for (Completion& c : sched->TakeCompleted()) {
    out.completions.push_back(std::move(c));
  }

  return out;
}

// The point of §6.3: the second request for a prompt does not recompute the
// part of it the first already did.
TEST_F(SchedulerTest, ASecondRequestSkipsThePrefixItWouldHaveRecomputed) {
  // 16 tokens is 4 blocks at this block size.
  std::vector<int32_t> prompt(16);
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = static_cast<int32_t>(i);

  ASSERT_TRUE(sched_->AddRequest(1, prompt, Params(2)).ok());
  const Workload cold = RunWorkload(sched_.get());
  ASSERT_EQ(cold.completions.size(), 1u);

  EXPECT_EQ(sched_->prefix_hit_tokens(), 0) << "nothing was cached yet";
  EXPECT_GT(sched_->cached_blocks(), 0) << "the prompt was not offered";

  ASSERT_TRUE(sched_->AddRequest(2, prompt, Params(2)).ok());
  const Workload warm = RunWorkload(sched_.get());
  ASSERT_EQ(warm.completions.size(), 1u);

  // A whole prompt costs 16 forwarded tokens plus its generation; a warm one
  // starts 12 tokens in, the last block being left to compute.
  EXPECT_LT(warm.tokens_forwarded, cold.tokens_forwarded);
  EXPECT_EQ(cold.tokens_forwarded - warm.tokens_forwarded, 12);
  EXPECT_EQ(sched_->prefix_hit_tokens(), 12);

  // And the tokens are the same ones, which is the part that has to be true
  // for any of this to be allowed.
  EXPECT_EQ(warm.completions[0].output_tokens,
            cold.completions[0].output_tokens);
}

// The property the whole feature has to clear: caching is invisible in the
// output. A shared prefix is only reusable because a key depends on the tokens
// before it, and the path through the tree is exactly those tokens -- get that
// wrong and a request silently reads someone else's history.
TEST_F(SchedulerTest, TheCacheDoesNotChangeWhatIsGenerated) {
  // Overlapping prompts: shared heads, divergent tails, some repeated exactly.
  const auto build = [](int32_t family, int32_t tail) {
    std::vector<int32_t> p;
    for (int32_t i = 0; i < 12; ++i) p.push_back(family * 10 + i);
    for (int32_t i = 0; i < 6; ++i) p.push_back(tail * 100 + i);
    return p;
  };

  const auto run = [&](bool enable_cache) {
    auto pool = HostPool(128);
    EXPECT_TRUE(pool.ok());
    auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

    SchedulerConfig config;
    config.max_running = 3;
    config.max_batch_tokens = 64;
    config.max_seq_len = 64;
    config.enable_prefix_cache = enable_cache;

    auto created = Scheduler::Create(config, owned.get());
    EXPECT_TRUE(created.ok());
    auto sched = std::make_unique<Scheduler>(*std::move(created));

    std::vector<Completion> all;
    RequestId id = 1;

    // Submitted in waves, so later waves meet a populated cache.
    for (int wave = 0; wave < 3; ++wave) {
      for (int32_t family = 0; family < 2; ++family) {
        for (int32_t tail = 0; tail < 2; ++tail) {
          EXPECT_TRUE(
              sched->AddRequest(id++, build(family, tail), Params(4)).ok());
        }
      }

      Workload w = RunWorkload(sched.get());
      for (Completion& c : w.completions) all.push_back(std::move(c));
    }

    ExpectNothingLeaked(*owned, *sched);

    std::sort(all.begin(), all.end(),
              [](const Completion& a, const Completion& b) {
                return a.id < b.id;
              });
    return all;
  };

  const std::vector<Completion> without = run(/*enable_cache=*/false);
  const std::vector<Completion> with = run(/*enable_cache=*/true);

  ASSERT_EQ(with.size(), without.size());
  ASSERT_EQ(with.size(), 12u);

  for (size_t i = 0; i < with.size(); ++i) {
    EXPECT_EQ(with[i].id, without[i].id);
    EXPECT_EQ(with[i].reason, without[i].reason) << "request " << with[i].id;
    EXPECT_EQ(with[i].output_tokens, without[i].output_tokens)
        << "request " << with[i].id << " changed when the cache was enabled";
  }
}

// §8.2's bet, now collectable. A preempted sequence's history goes into the
// tree rather than being discarded, so when it is re-admitted it matches its
// own prefix and the recompute preemption budgets for does not happen.
//
// This is what T7 chose recompute over swapping for, and until §6.3 landed it
// was an argument rather than a behaviour.
TEST_F(SchedulerTest, APreemptedSequenceGetsItsOwnHistoryBackFromTheCache) {
  auto pool = HostPool(/*blocks=*/10);
  ASSERT_TRUE(pool.ok());
  auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

  SchedulerConfig config;
  config.max_running = 3;
  config.max_batch_tokens = 64;
  config.max_seq_len = 64;

  auto created = Scheduler::Create(config, owned.get());
  ASSERT_TRUE(created.ok());
  auto sched = std::make_unique<Scheduler>(*std::move(created));

  // Long-ish prompts against a pool that cannot hold all three growing.
  for (int i = 0; i < 3; ++i) {
    std::vector<int32_t> prompt(12);
    for (size_t k = 0; k < prompt.size(); ++k) {
      prompt[k] = static_cast<int32_t>(i * 100 + k);
    }
    ASSERT_TRUE(sched->AddRequest(static_cast<RequestId>(i + 1), prompt,
                                  Params(12))
                    .ok());
  }

  const Workload w = RunWorkload(sched.get());

  ASSERT_EQ(w.completions.size(), 3u);
  ASSERT_GT(sched->preemptions(), 0) << "nothing was preempted";

  // The re-admitted sequences found their own prefixes waiting. Without the
  // cache every preemption would be a prompt prefilled from scratch again.
  EXPECT_GT(sched->prefix_hit_tokens(), 0)
      << "a preempted sequence recomputed what it had already computed";

  ExpectNothingLeaked(*owned, *sched);
}

// The cache must not make the engine worse at running out of memory. Cached
// blocks are reclaimable, so a pool that looks full because of them still
// admits work rather than stalling.
TEST_F(SchedulerTest, ACacheFullOfBlocksStillYieldsToARunningRequest) {
  auto pool = HostPool(/*blocks=*/4);
  ASSERT_TRUE(pool.ok());
  auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

  SchedulerConfig config;
  config.max_running = 2;
  config.max_batch_tokens = 64;
  config.max_seq_len = 64;

  auto created = Scheduler::Create(config, owned.get());
  ASSERT_TRUE(created.ok());
  auto sched = std::make_unique<Scheduler>(*std::move(created));

  // Fill the cache with a prompt nobody will ask for again.
  ASSERT_TRUE(sched->AddRequest(1, {9, 9, 9, 9, 9, 9, 9, 9}, Params(1)).ok());
  ASSERT_EQ(RunWorkload(sched.get()).completions.size(), 1u);

  ASSERT_GT(sched->cached_blocks(), 0);
  ASSERT_EQ(owned->free_blocks(), owned->num_blocks() - sched->cached_blocks());

  // An unrelated request now needs that memory. The allocator evicts to get it
  // rather than reporting the pool full.
  ASSERT_TRUE(sched->AddRequest(2, {1, 2, 3, 4, 5, 6, 7, 8}, Params(4)).ok());

  const Workload w = RunWorkload(sched.get());

  ASSERT_EQ(w.completions.size(), 1u);
  EXPECT_EQ(w.completions[0].reason, FinishReason::kMaxTokens)
      << "the cache starved a request that should have evicted it";
  EXPECT_EQ(w.completions[0].output_tokens.size(), 4u);
}

// Once the scheduler is gone, so is every claim on the pool -- including the
// cache's. Checked after destruction rather than during, because that is the
// only point at which the answer is unambiguously "all of them": while the
// scheduler lives, blocks legitimately sit in the radix tree.
TEST_F(SchedulerTest, DestroyingAPressuredSchedulerReturnsEveryBlock) {
  auto pool = HostPool(/*blocks=*/16, /*block_size=*/16);
  ASSERT_TRUE(pool.ok());
  auto owned = std::make_unique<KvBlockPool>(*std::move(pool));

  {
    SchedulerConfig config;
    config.max_running = 4;
    config.max_batch_tokens = 256;
    config.max_seq_len = 512;

    auto created = Scheduler::Create(config, owned.get());
    ASSERT_TRUE(created.ok());
    Scheduler sched = *std::move(created);

    // Four copies of one prompt against a pool that cannot hold them growing,
    // so the run preempts, evicts, and matches all at once.
    std::vector<int32_t> prompt(40);
    for (size_t i = 0; i < prompt.size(); ++i) {
      prompt[i] = static_cast<int32_t>(i % 4);
    }

    for (int i = 0; i < 4; ++i) {
      ASSERT_TRUE(
          sched.AddRequest(static_cast<RequestId>(i + 1), prompt, Params(48))
              .ok());
    }

    const Workload w = RunWorkload(&sched);

    EXPECT_EQ(w.completions.size(), 4u) << "a request was lost";
    EXPECT_GT(sched.preemptions(), 0) << "the pool was not tight enough";
  }

  EXPECT_EQ(owned->free_blocks(), owned->num_blocks())
      << "the scheduler and its cache did not give everything back";
}

// A scheduler destroyed with requests still in flight gives their blocks back.
//
// `BlockTable` holds no pointer to the pool it drew from, so it cannot return
// its own blocks and nothing below the scheduler can either. Until the
// scheduler's destructor did it, a server shut down mid-stream stranded every
// in-flight sequence's KV in a pool that outlived it.
TEST_F(SchedulerTest, DestructionReturnsTheBlocksOfUnfinishedRequests) {
  const int64_t total = pool_->num_blocks();

  {
    auto sched = ChunkingScheduler(pool_.get(), /*max_batch_tokens=*/64);
    ASSERT_NE(sched, nullptr);

    // Enough requests to exceed max_running, so some are still queued when the
    // scheduler goes away and others are running with blocks held.
    for (int i = 0; i < 6; ++i) {
      ASSERT_TRUE(sched->AddRequest(static_cast<RequestId>(i + 1),
                                    {1, 2, 3, 4, 5}, Params(50))
                      .ok());
    }

    ForwardBatch batch;
    ASSERT_TRUE(sched->PrepareStep(&batch).ok());
    ASSERT_LT(pool_->free_blocks(), total) << "nothing was allocated to leak";
  }

  EXPECT_EQ(pool_->free_blocks(), total)
      << "a scheduler destroyed mid-flight stranded its blocks";
}

TEST_F(SchedulerTest, CommitRejectsTheWrongNumberOfTokens) {
  ASSERT_TRUE(sched_->AddRequest(1, {1, 2}, Params(2)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

  EXPECT_EQ(sched_->CommitStep({1, 2, 3}).code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(SchedulerTest, RejectsMalformedRequests) {
  EXPECT_EQ(sched_->AddRequest(1, {}, Params(4)).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(sched_->AddRequest(1, {1}, Params(0)).code(),
            absl::StatusCode::kInvalidArgument);
}

// The full loop, driven to completion with a fake executor. This is §5.2's
// depth-0 path -- prepare, execute, commit, nothing in flight -- which T6 keeps
// as the reference the overlapped version gets diffed against.
TEST_F(SchedulerTest, DrainsAMixedWorkloadToCompletion) {
  ASSERT_TRUE(sched_->AddRequest(1, {1, 2, 3}, Params(2)).ok());
  ASSERT_TRUE(sched_->AddRequest(2, {4}, Params(5, {77})).ok());
  ASSERT_TRUE(sched_->AddRequest(3, {5, 6}, Params(3)).ok());

  std::vector<Completion> all;
  int32_t next_token = 70;
  int steps = 0;

  while (sched_->HasWork()) {
    ASSERT_LT(++steps, 100) << "step loop did not terminate";

    ForwardBatch batch;
    ASSERT_TRUE(sched_->PrepareStep(&batch).ok());

    if (batch.num_tokens() == 0) {
      for (Completion& c : sched_->TakeCompleted()) all.push_back(std::move(c));
      continue;
    }

    ASSERT_TRUE(batch.Validate(1000, pool_->num_blocks() * pool_->block_size())
                    .ok())
        << "step " << steps << " produced an invalid batch";

    // Stand-in for the executor: hand back one token per requested logits row.
    std::vector<int32_t> sampled(batch.logits_indices.size());
    for (int32_t& t : sampled) t = next_token++;

    ASSERT_TRUE(sched_->CommitStep(sampled).ok());

    for (Completion& c : sched_->TakeCompleted()) all.push_back(std::move(c));
  }

  for (Completion& c : sched_->TakeCompleted()) all.push_back(std::move(c));

  ASSERT_EQ(all.size(), 3u);
  for (const Completion& c : all) {
    EXPECT_NE(c.reason, FinishReason::kNotFinished) << "request " << c.id;
    EXPECT_FALSE(c.output_tokens.empty()) << "request " << c.id;
  }

  ExpectNothingLeaked(*pool_, *sched_);
}

}  // namespace

// Every position in the batch must be reachable through the batch's own block
// table.
//
// This is R8, and it is the invariant the whole paged design rests on: `slots`
// says where a token's key and value are written, the block table says where
// attention looks for them, and the two are only the same memory if the table
// covers every block the slots touch. They diverged on exactly one step -- the
// one where a sequence first grows into a new block -- because the row was
// copied out before the growth happened.
//
// The consequence was not a crash. The uncovered entry reads zero, so attention
// silently attended to block 0, and whether that was wrong depended on which
// physical block the allocator had handed out. Consecutive requests got their
// blocks in opposite order from the stack free list, so the same prompt came
// back with two different answers in alternation.
//
// Checked here, on the host, with no device anywhere near it -- which is the
// argument for §3.1's split. The bug lived in the scheduler the whole time, and
// it took a GPU, a model and an HTTP server to notice.
TEST_F(SchedulerTest, EveryBatchSlotIsCoveredByTheBatchBlockTable) {
  const int64_t block_size = pool_->block_size();

  ASSERT_TRUE(sched_->AddRequest(1, {1, 2, 3}, Params(/*max_tokens=*/50)).ok());

  ForwardBatch batch;

  // Long enough to cross several block boundaries, which is where the two
  // disagreed.
  for (int step = 0; step < 60; ++step) {
    ASSERT_TRUE(sched_->PrepareStep(&batch).ok()) << "step " << step;

    if (batch.token_ids.empty()) break;

    for (size_t i = 0; i < batch.slots.size(); ++i) {
      const int32_t slot = batch.slots[i];
      const int64_t position = batch.positions[i];
      const int64_t seq = batch.seq_of_token[i];

      // Where the token's KV is actually written.
      const int32_t physical_block =
          static_cast<int32_t>(slot / block_size);

      // Where attention will look for it: the logical block this position
      // falls in, resolved through the batch's own table.
      const int64_t logical_block = position / block_size;

      ASSERT_LT(logical_block, batch.max_blocks_per_seq)
          << "step " << step << ": position " << position
          << " is past the table's width";

      const int32_t via_table = batch.block_table[static_cast<size_t>(
          seq * batch.max_blocks_per_seq + logical_block)];

      EXPECT_EQ(via_table, physical_block)
          << "step " << step << ": token at position " << position
          << " is written to block " << physical_block
          << " but the batch's block table sends attention to block "
          << via_table
          << " -- the KV written this step is not the KV that will be read";
    }

    std::vector<int32_t> sampled(batch.logits_indices.size(), 7);
    ASSERT_TRUE(sched_->CommitStep(sampled).ok()) << "step " << step;
  }
}

}  // namespace inferx::scheduler
