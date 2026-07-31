// The scheduler's whole lifecycle, on a machine with no GPU.
//
// That this file is in tests/unit rather than tests/kernel is the point. §3.1
// calls the scheduler/executor split the highest-leverage testability decision
// in the design, and this is what it buys: admission, block accounting,
// batching, stop conditions, cancellation and out-of-memory are all exercised
// against a host-allocated pool, with no device present and no forward pass
// involved. A bug in any of them is found here rather than inside a 6 GB model.

#include "inferx/scheduler/scheduler.h"

#include <vector>

#include <gtest/gtest.h>

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

  // Finish the first; its blocks come back and the second is admitted.
  ASSERT_TRUE(sched.CommitStep({9}).ok());
  EXPECT_EQ(pool.free_blocks(), 2) << "blocks leaked on the deferred admission";

  ASSERT_TRUE(sched.PrepareStep(&batch).ok());
  EXPECT_EQ(sched.num_running(), 1);
  EXPECT_EQ(batch.token_ids, (std::vector<int32_t>{2, 2, 2, 2}));
}

// A prefill larger than the per-step token budget waits for a step of its own
// rather than being truncated. Chunking it is M5's job; silently running part
// of a prompt would be a correctness bug, not a performance one.
TEST_F(SchedulerTest, OversizedPrefillDefersRatherThanTruncating) {
  SchedulerConfig config;
  config.max_running = 4;
  config.max_batch_tokens = 4;
  config.max_seq_len = 64;

  auto s = Scheduler::Create(config, pool_.get());
  ASSERT_TRUE(s.ok());
  Scheduler sched = *std::move(s);

  ASSERT_TRUE(sched.AddRequest(1, {1, 2, 3}, Params(2)).ok());
  ASSERT_TRUE(sched.AddRequest(2, {4, 5, 6}, Params(2)).ok());

  ForwardBatch batch;
  ASSERT_TRUE(sched.PrepareStep(&batch).ok());

  // The first prefill takes 3 of 4; the second needs 3 and does not fit, so it
  // contributes nothing this step -- but it is not cut short.
  EXPECT_EQ(batch.token_ids, (std::vector<int32_t>{1, 2, 3}));
  EXPECT_EQ(batch.logits_indices.size(), 1u);

  ASSERT_TRUE(sched.CommitStep({7}).ok());

  ASSERT_TRUE(sched.PrepareStep(&batch).ok());
  // Now the second prefill runs, alongside the first's decode.
  EXPECT_EQ(batch.token_ids, (std::vector<int32_t>{7, 4, 5, 6}));
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

  EXPECT_EQ(sched_->blocks_in_use(), 0) << "blocks leaked across the workload";
  EXPECT_EQ(pool_->free_blocks(), pool_->num_blocks());
}

}  // namespace
}  // namespace inferx::scheduler
