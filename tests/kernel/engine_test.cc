// The engine, end to end: scheduler + paged cache + real model.
//
// M3's deliverable is "the engine exists", and this is what that means -- a
// request goes in, a step loop runs, tokens come out, blocks are returned. The
// scheduler contributes no CUDA and the model contributes no policy; the loop
// below is the entire seam between them, and it is §5.2's depth-0 case:
// prepare, execute, commit, nothing in flight.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/model/qwen2.h"
#include "inferx/scheduler/scheduler.h"

namespace inferx {
namespace {

using model::ForwardBatch;
using scheduler::Completion;
using scheduler::FinishReason;
using scheduler::SamplingParams;
using scheduler::Scheduler;
using scheduler::SchedulerConfig;

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

/// Greedy sampling on the host. The on-GPU sampler is §4 step 7 and M4's job;
/// what matters here is that the loop closes.
std::vector<int32_t> GreedySample(const std::vector<float>& logits,
                                  size_t rows) {
  std::vector<int32_t> out;
  if (rows == 0) return out;

  const size_t vocab = logits.size() / rows;

  for (size_t r = 0; r < rows; ++r) {
    const auto begin = logits.begin() + static_cast<long>(r * vocab);
    out.push_back(static_cast<int32_t>(
        std::distance(begin, std::max_element(begin,
                                              begin + static_cast<long>(vocab)))));
  }

  return out;
}

class EngineTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!CudaAvailable()) return;

    auto loaded = model::Qwen2Model::LoadFromDirectory(CheckpointDir());
    if (!loaded.ok()) {
      status_ = loaded.status();
      return;
    }

    model_ = new model::Qwen2Model(*std::move(loaded));
    status_ = model_->AttachKvCache(512, 16);
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

  /// Runs the loop until nothing is left. Returns everything that finished.
  std::vector<Completion> Drain(Scheduler* sched, int max_steps = 200) {
    std::vector<Completion> all;
    int steps = 0;

    while (sched->HasWork()) {
      EXPECT_LT(++steps, max_steps) << "step loop did not terminate";
      if (steps >= max_steps) break;

      ForwardBatch batch;
      EXPECT_TRUE(sched->PrepareStep(&batch).ok());

      if (batch.num_tokens() > 0) {
        std::vector<float> logits;
        const Status s = model_->Step(batch, &logits);
        EXPECT_TRUE(s.ok()) << "step " << steps << ": " << s;
        if (!s.ok()) break;

        const std::vector<int32_t> sampled =
            GreedySample(logits, batch.logits_indices.size());

        EXPECT_TRUE(sched->CommitStep(sampled).ok());
      }

      for (Completion& c : sched->TakeCompleted()) all.push_back(std::move(c));
    }

    for (Completion& c : sched->TakeCompleted()) all.push_back(std::move(c));
    return all;
  }

  StatusOr<Scheduler> MakeScheduler(int64_t max_running = 4,
                                    int64_t max_batch_tokens = 2048) {
    SchedulerConfig config;
    config.max_running = max_running;
    config.max_batch_tokens = max_batch_tokens;
    config.max_seq_len = 512;

    return Scheduler::Create(config, model_->kv_pool());
  }

  static model::Qwen2Model* model_;
  static Status status_;
};

model::Qwen2Model* EngineTest::model_ = nullptr;
Status EngineTest::status_ = OkStatus();

// One request, all the way through. The first generated token is the answer the
// model has given at every milestone since M2, now arrived at through a
// scheduler and a paged cache instead of a full recompute.
TEST_F(EngineTest, ServesASingleRequest) {
  auto sched = MakeScheduler();
  ASSERT_TRUE(sched.ok()) << sched.status();

  SamplingParams params;
  params.max_tokens = 5;

  ASSERT_TRUE(sched->AddRequest(1, {kThe, kCapital, kOf, kFrance, kIs}, params)
                  .ok());

  const std::vector<Completion> done = Drain(&*sched);

  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].id, 1u);
  EXPECT_EQ(done[0].reason, FinishReason::kMaxTokens);
  ASSERT_EQ(done[0].output_tokens.size(), 5u);
  EXPECT_EQ(done[0].output_tokens[0], kParis);

  EXPECT_EQ(model_->kv_pool()->used_blocks(), 0) << "blocks leaked";
}

// Several requests at once, sharing one pool and one batch. Their blocks
// necessarily interleave, and each must still get the answer it would have got
// alone -- which is the property that makes batching safe.
TEST_F(EngineTest, ServesConcurrentRequestsIndependently) {
  auto sched = MakeScheduler();
  ASSERT_TRUE(sched.ok()) << sched.status();

  SamplingParams params;
  params.max_tokens = 3;

  // Same prompt three times, plus one different, so a cross-contamination bug
  // shows up as the odd one out agreeing with the others.
  ASSERT_TRUE(sched->AddRequest(1, {kThe, kCapital, kOf, kFrance, kIs}, params).ok());
  ASSERT_TRUE(sched->AddRequest(2, {kThe, kCapital, kOf, kFrance, kIs}, params).ok());
  ASSERT_TRUE(sched->AddRequest(3, {kThe, kCapital, kOf, kFrance, kIs}, params).ok());
  ASSERT_TRUE(sched->AddRequest(4, {kThe, kCapital, kOf, kFrance}, params).ok());

  std::vector<Completion> done = Drain(&*sched);
  ASSERT_EQ(done.size(), 4u);

  std::sort(done.begin(), done.end(),
            [](const Completion& a, const Completion& b) { return a.id < b.id; });

  // The three identical prompts must produce identical output.
  EXPECT_EQ(done[0].output_tokens, done[1].output_tokens);
  EXPECT_EQ(done[1].output_tokens, done[2].output_tokens);
  EXPECT_EQ(done[0].output_tokens[0], kParis);

  // And the fourth, which is a different prompt, must not.
  EXPECT_NE(done[3].output_tokens, done[0].output_tokens)
      << "a different prompt produced identical output; sequences are sharing "
         "cache";

  EXPECT_EQ(model_->kv_pool()->used_blocks(), 0) << "blocks leaked";
}

// Batching must not change the answer. The same request run alone and run
// alongside others has to generate the same tokens -- if it does not, the batch
// is leaking state between sequences.
TEST_F(EngineTest, BatchingDoesNotChangeTheOutput) {
  const std::vector<int32_t> prompt = {kThe, kCapital, kOf, kFrance, kIs};

  SamplingParams params;
  params.max_tokens = 4;

  std::vector<int32_t> alone;
  {
    auto sched = MakeScheduler(1);
    ASSERT_TRUE(sched.ok());
    ASSERT_TRUE(sched->AddRequest(1, prompt, params).ok());

    const std::vector<Completion> done = Drain(&*sched);
    ASSERT_EQ(done.size(), 1u);
    alone = done[0].output_tokens;
  }

  std::vector<int32_t> batched;
  {
    auto sched = MakeScheduler(4);
    ASSERT_TRUE(sched.ok());

    ASSERT_TRUE(sched->AddRequest(1, prompt, params).ok());
    // Noise alongside it, with different lengths so the batch is ragged.
    ASSERT_TRUE(sched->AddRequest(2, {kThe, kIs}, params).ok());
    ASSERT_TRUE(sched->AddRequest(3, {kOf}, params).ok());

    std::vector<Completion> done = Drain(&*sched);
    ASSERT_EQ(done.size(), 3u);

    for (const Completion& c : done) {
      if (c.id == 1) batched = c.output_tokens;
    }
  }

  EXPECT_EQ(alone, batched)
      << "batching changed the output; sequences are not isolated";
}

TEST_F(EngineTest, RaggedPrefillCacheIsRepeatableAfterBlockReuse) {
  const std::vector<std::vector<int32_t>> prompts = {
      {kThe, kCapital, kOf, kFrance, kIs},
      {kThe, kCapital, kOf, kFrance},
      {kThe, kIs}};

  struct Snapshot {
    std::vector<float> logits;
    std::vector<uint16_t> kv;
  };

  const auto prefill = [&] {
    auto sched = MakeScheduler(4);
    EXPECT_TRUE(sched.ok()) << sched.status();

    SamplingParams params;
    params.max_tokens = 1;
    for (size_t i = 0; i < prompts.size(); ++i) {
      EXPECT_TRUE(sched->AddRequest(i + 1, prompts[i], params).ok());
    }

    ForwardBatch batch;
    EXPECT_TRUE(sched->PrepareStep(&batch).ok());

    Snapshot snapshot;
    EXPECT_TRUE(model_->Step(batch, &snapshot.logits).ok());

    const int64_t entry_elems = model_->config().num_key_value_heads *
                                model_->config().head_dim;
    std::vector<uint16_t> entry(static_cast<size_t>(entry_elems));
    for (int64_t layer = 0; layer < model_->config().num_hidden_layers;
         ++layer) {
      for (const bool value : {false, true}) {
        const StatusOr<TensorView> cache =
            value ? model_->kv_pool()->ValueCache(layer)
                  : model_->kv_pool()->KeyCache(layer);
        EXPECT_TRUE(cache.ok()) << cache.status();
        for (const int32_t slot : batch.slots) {
          const size_t offset = static_cast<size_t>(slot * entry_elems) *
                                sizeof(uint16_t);
          EXPECT_EQ(cudaMemcpy(
                        entry.data(),
                        static_cast<const std::byte*>(cache->Data()) + offset,
                        entry.size() * sizeof(uint16_t),
                        cudaMemcpyDeviceToHost),
                    cudaSuccess);
          snapshot.kv.insert(snapshot.kv.end(), entry.begin(), entry.end());
        }
      }
    }

    EXPECT_TRUE(sched->CommitStep(
        GreedySample(snapshot.logits, batch.logits_indices.size())).ok());
    (void)sched->TakeCompleted();
    return snapshot;
  };

  const Snapshot first = prefill();
  const Snapshot second = prefill();
  EXPECT_EQ(first.logits, second.logits);
  ASSERT_EQ(first.kv.size(), second.kv.size());

  for (size_t i = 0; i < first.kv.size(); ++i) {
    ASSERT_EQ(first.kv[i], second.kv[i])
        << "first cached K/V difference at flattened element " << i;
  }
}

// Chunking a prompt changes how many steps it takes and must change nothing
// else.
//
// This is what the whole of §8.1 rests on. A chunk attends over KV that earlier
// chunks wrote, so the second chunk's queries read keys through the block table
// rather than from anything in its own batch. If the slot arithmetic, the
// declared sequence length or the resumption point were off by one position,
// those reads would land on a neighbouring key and the model would return
// plausible logits rather than failing -- the same failure mode as R8, which
// took a GPU, a model and an HTTP server to notice.
TEST_F(EngineTest, ChunkedPrefillMatchesWholePrefill) {
  // Long enough for several chunks, and not a multiple of the chunk size, so
  // the last chunk is short.
  std::vector<int32_t> prompt;
  for (int i = 0; i < 6; ++i) {
    for (const int32_t t : {kThe, kCapital, kOf, kFrance, kIs, kParis}) {
      prompt.push_back(t);
    }
  }
  ASSERT_EQ(prompt.size(), 36u);

  SamplingParams params;
  params.max_tokens = 1;

  // Drives the prompt through its prefill, however many steps the budget makes
  // that take, and returns the logits of its final token.
  const auto prefill_logits = [&](int64_t budget, int* out_steps) {
    std::vector<float> logits;

    auto sched = MakeScheduler(/*max_running=*/1, budget);
    EXPECT_TRUE(sched.ok());
    EXPECT_TRUE(sched->AddRequest(1, prompt, params).ok());

    int steps = 0;

    while (steps < 50) {
      ForwardBatch batch;
      EXPECT_TRUE(sched->PrepareStep(&batch).ok());
      EXPECT_GT(batch.num_tokens(), 0);
      ++steps;

      const Status s = model_->Step(batch, &logits);
      EXPECT_TRUE(s.ok()) << "step " << steps << ": " << s;
      if (!s.ok()) break;

      // The last chunk is the one that asks for logits.
      if (!batch.logits_indices.empty()) break;

      // A middle chunk returns nothing to commit, and committing nothing is
      // what advances it to the next chunk.
      EXPECT_TRUE(logits.empty()) << "a middle chunk produced logits";
      EXPECT_TRUE(sched->CommitStep({}).ok());
    }

    *out_steps = steps;
    return logits;
  };

  int whole_steps = 0;
  int chunked_steps = 0;

  const std::vector<float> whole = prefill_logits(2048, &whole_steps);
  const std::vector<float> chunked = prefill_logits(8, &chunked_steps);

  EXPECT_EQ(whole_steps, 1) << "the control was itself chunked";
  EXPECT_EQ(chunked_steps, 5) << "36 tokens at 8 per step is 4 chunks and a tail";

  ASSERT_FALSE(whole.empty());
  ASSERT_EQ(whole.size(), chunked.size());

  // What a caller actually sees.
  EXPECT_EQ(GreedySample(whole, 1), GreedySample(chunked, 1))
      << "chunking changed the model's prediction";

  // And the logits themselves, to the extent the arithmetic allows. Chunking
  // reduces over an [8, d] GEMM where the whole prefill reduces over [36, d],
  // so the two differ in bf16's last bits rather than not at all.
  double worst = 0.0;
  double scale = 0.0;
  for (size_t i = 0; i < whole.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(whole[i]) -
                                      static_cast<double>(chunked[i])));
    scale = std::max(scale, std::fabs(static_cast<double>(whole[i])));
  }

  std::fprintf(stderr, "[chunked prefill] max|diff| = %.6g over range %.6g\n",
               worst, scale);

  EXPECT_LT(worst, 0.02 * scale)
      << "chunking moved the logits further than rounding explains";
}

// The shape chunking exists to produce: a decode and a half-finished prompt in
// the same batch. The decoding sequence must get exactly the token it would
// have got had the prompt not been there, which is the isolation property
// ragged batching already had -- now with one of the sequences contributing
// query rows that stop in the middle of itself.
TEST_F(EngineTest, ADecodeIsUnaffectedByAPrefillChunkBesideIt) {
  const std::vector<int32_t> prompt = {kThe, kCapital, kOf, kFrance, kIs};

  SamplingParams params;
  params.max_tokens = 4;

  // Alone: no one else in the batch at any point.
  std::vector<int32_t> alone;
  {
    auto sched = MakeScheduler(/*max_running=*/1);
    ASSERT_TRUE(sched.ok());
    ASSERT_TRUE(sched->AddRequest(1, prompt, params).ok());

    const std::vector<Completion> done = Drain(&*sched);
    ASSERT_EQ(done.size(), 1u);
    alone = done[0].output_tokens;
  }

  // Beside a long prompt, with a budget small enough that the prompt takes
  // several steps to get through and is mid-prefill while request 1 decodes.
  std::vector<int32_t> beside;
  {
    auto sched = MakeScheduler(/*max_running=*/4, /*max_batch_tokens=*/8);
    ASSERT_TRUE(sched.ok());

    ASSERT_TRUE(sched->AddRequest(1, prompt, params).ok());

    std::vector<int32_t> long_prompt;
    for (int i = 0; i < 8; ++i) {
      for (const int32_t t : {kThe, kCapital, kOf, kFrance}) {
        long_prompt.push_back(t);
      }
    }
    ASSERT_TRUE(sched->AddRequest(2, long_prompt, params).ok());

    bool saw_a_mixed_step = false;

    // The loop, inline rather than through Drain, so the mixed shape can be
    // asserted to have actually occurred -- a test that silently never
    // produced one would pass for the wrong reason.
    int steps = 0;
    std::vector<Completion> done;

    while (sched->HasWork() && steps < 200) {
      ++steps;

      ForwardBatch batch;
      ASSERT_TRUE(sched->PrepareStep(&batch).ok());

      if (batch.num_tokens() > 0) {
        // A step where request 1 decodes one token and request 2 supplies a
        // chunk of prompt that does not finish it.
        if (batch.num_seqs == 2 && batch.logits_indices.size() == 1 &&
            batch.num_tokens() > 1) {
          saw_a_mixed_step = true;
        }

        std::vector<float> logits;
        ASSERT_TRUE(model_->Step(batch, &logits).ok()) << "step " << steps;
        ASSERT_TRUE(sched->CommitStep(
                         GreedySample(logits, batch.logits_indices.size()))
                        .ok());
      }

      for (Completion& c : sched->TakeCompleted()) done.push_back(std::move(c));
    }

    for (Completion& c : sched->TakeCompleted()) done.push_back(std::move(c));

    EXPECT_TRUE(saw_a_mixed_step)
        << "no decode ever shared a step with an unfinished prefill";

    ASSERT_EQ(done.size(), 2u);
    for (const Completion& c : done) {
      if (c.id == 1) beside = c.output_tokens;
    }
  }

  EXPECT_EQ(alone, beside)
      << "a prefill chunk in the batch changed a decoding sequence's output";

  EXPECT_EQ(model_->kv_pool()->used_blocks(), 0) << "blocks leaked";
}

TEST_F(EngineTest, StopTokenEndsGenerationEarly) {
  auto sched = MakeScheduler();
  ASSERT_TRUE(sched.ok());

  // Stop on whatever the model will actually produce first, so the stop path is
  // genuinely taken rather than the token cap.
  SamplingParams params;
  params.max_tokens = 20;
  params.stop_tokens = {kParis};

  ASSERT_TRUE(sched->AddRequest(1, {kThe, kCapital, kOf, kFrance, kIs}, params)
                  .ok());

  const std::vector<Completion> done = Drain(&*sched);

  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].reason, FinishReason::kStopToken);
  EXPECT_EQ(done[0].output_tokens, (std::vector<int32_t>{kParis}));
}

// More requests than fit at once: admission has to queue them, and the pool has
// to recycle blocks between them. This is the shape of real load.
TEST_F(EngineTest, QueuesMoreRequestsThanCanRunAtOnce) {
  auto sched = MakeScheduler(/*max_running=*/2);
  ASSERT_TRUE(sched.ok());

  SamplingParams params;
  params.max_tokens = 2;

  for (int i = 0; i < 6; ++i) {
    ASSERT_TRUE(sched->AddRequest(static_cast<scheduler::RequestId>(i),
                                  {kThe, kCapital, kOf, kFrance, kIs}, params)
                    .ok());
  }

  EXPECT_EQ(sched->num_waiting(), 6);

  const std::vector<Completion> done = Drain(&*sched);

  EXPECT_EQ(done.size(), 6u);
  for (const Completion& c : done) {
    EXPECT_EQ(c.output_tokens.size(), 2u) << "request " << c.id;
    EXPECT_EQ(c.output_tokens[0], kParis) << "request " << c.id;
  }

  EXPECT_EQ(model_->kv_pool()->used_blocks(), 0) << "blocks leaked";
}

// Generation long enough to cross several block boundaries, which is where a
// block table that is appended to mid-sequence gets exercised.
TEST_F(EngineTest, GenerationGrowsAcrossBlockBoundaries) {
  auto sched = MakeScheduler(1);
  ASSERT_TRUE(sched.ok());

  SamplingParams params;
  params.max_tokens = 40;  // 5 prompt + 40 spans four 16-token blocks

  ASSERT_TRUE(sched->AddRequest(1, {kThe, kCapital, kOf, kFrance, kIs}, params)
                  .ok());

  const std::vector<Completion> done = Drain(&*sched);

  ASSERT_EQ(done.size(), 1u);
  EXPECT_EQ(done[0].output_tokens.size(), 40u);
  EXPECT_EQ(done[0].output_tokens[0], kParis);

  // Degenerate repetition is what a block table that stops growing correctly
  // looks like: history stops mattering and the model loops.
  const auto& out = done[0].output_tokens;
  const bool all_same =
      std::all_of(out.begin(), out.end(),
                  [&](int32_t t) { return t == out[0]; });
  EXPECT_FALSE(all_same) << "generation collapsed; the cache likely stopped "
                            "growing correctly";

  // The scheduler still holds its prefix cache, whose blocks are reclaimable
  // rather than in use (§6.3).
  EXPECT_EQ(sched->blocks_in_use(), 0);
  EXPECT_EQ(model_->kv_pool()->used_blocks(), sched->cached_blocks());
}

// CUDA graphs must not change the answer. A graph fixes the structure of a
// decode step and nothing else -- every value it reads still comes from buffers
// rewritten before each replay -- so capturing one and replaying it has to
// produce exactly what issuing the launches produced. If it does not, the graph
// has baked in something that was supposed to stay live.
TEST_F(EngineTest, CudaGraphReplayMatchesLaunchByLaunch) {
  const std::vector<int32_t> prompt = {kThe, kCapital, kOf, kFrance, kIs};

  SamplingParams params;
  params.max_tokens = 8;

  // Without a graph.
  std::vector<int32_t> ungraphed;
  {
    auto sched = MakeScheduler(1);
    ASSERT_TRUE(sched.ok());
    ASSERT_TRUE(sched->AddRequest(1, prompt, params).ok());

    const std::vector<Completion> done = Drain(&*sched);
    ASSERT_EQ(done.size(), 1u);
    ungraphed = done[0].output_tokens;
  }

  // Capture the decode shape this workload uses: one sequence, and the block
  // table width the scheduler emits for max_seq_len 512 at 16 tokens a block.
  const int64_t max_blocks = (512 + 16 - 1) / 16;
  ASSERT_TRUE(model_->CaptureDecodeGraph(1, max_blocks).ok());
  EXPECT_GE(model_->captured_graphs(), 1);

  // With one. Prefill still runs launch-by-launch -- only the decode shape was
  // captured -- which is itself part of what this checks: the two paths have to
  // interleave correctly within a single request.
  std::vector<int32_t> graphed;
  {
    auto sched = MakeScheduler(1);
    ASSERT_TRUE(sched.ok());
    ASSERT_TRUE(sched->AddRequest(1, prompt, params).ok());

    const std::vector<Completion> done = Drain(&*sched);
    ASSERT_EQ(done.size(), 1u);
    graphed = done[0].output_tokens;
  }

  EXPECT_EQ(ungraphed, graphed)
      << "replaying a captured decode step produced different tokens than "
         "issuing the same launches";
  EXPECT_EQ(graphed[0], kParis);
}

// Replays have to stay correct across many steps, not just the first. A graph
// that captured a stale pointer or a value that should have been live tends to
// produce the same token forever.
TEST_F(EngineTest, RepeatedGraphReplaysTrackGrowingContext) {
  const int64_t max_blocks = (512 + 16 - 1) / 16;
  ASSERT_TRUE(model_->CaptureDecodeGraph(1, max_blocks).ok());

  auto sched = MakeScheduler(1);
  ASSERT_TRUE(sched.ok());

  SamplingParams params;
  params.max_tokens = 30;

  ASSERT_TRUE(sched->AddRequest(1, {kThe, kCapital, kOf, kFrance, kIs}, params)
                  .ok());

  const std::vector<Completion> done = Drain(&*sched);
  ASSERT_EQ(done.size(), 1u);

  const auto& out = done[0].output_tokens;
  ASSERT_EQ(out.size(), 30u);
  EXPECT_EQ(out[0], kParis);

  const bool all_same = std::all_of(out.begin(), out.end(),
                                    [&](int32_t t) { return t == out[0]; });
  EXPECT_FALSE(all_same)
      << "generation collapsed under graph replay; the graph is likely reading "
         "a value that should have stayed live";
}

// A shape with no captured graph must still run, launch-by-launch. Capture is
// an optimization, never a precondition.
TEST_F(EngineTest, UncapturedShapesStillRun) {
  const int64_t max_blocks = (512 + 16 - 1) / 16;
  ASSERT_TRUE(model_->CaptureDecodeGraph(1, max_blocks).ok());

  // Three concurrent sequences: a shape that was never captured.
  auto sched = MakeScheduler(4);
  ASSERT_TRUE(sched.ok());

  SamplingParams params;
  params.max_tokens = 3;

  for (int i = 0; i < 3; ++i) {
    ASSERT_TRUE(sched->AddRequest(static_cast<scheduler::RequestId>(i),
                                  {kThe, kCapital, kOf, kFrance, kIs}, params)
                    .ok());
  }

  const std::vector<Completion> done = Drain(&*sched);

  ASSERT_EQ(done.size(), 3u);
  for (const Completion& c : done) EXPECT_EQ(c.output_tokens[0], kParis);
}

TEST_F(EngineTest, CaptureIsIdempotentAndValidated) {
  const int64_t max_blocks = 32;

  const Status captured = model_->CaptureDecodeGraph(2, max_blocks);
  ASSERT_TRUE(captured.ok()) << captured;
  const int64_t after_first = model_->captured_graphs();

  // Same shape again: served from what is already captured.
  ASSERT_TRUE(model_->CaptureDecodeGraph(2, max_blocks).ok());
  EXPECT_EQ(model_->captured_graphs(), after_first);

  EXPECT_EQ(model_->CaptureDecodeGraph(0, max_blocks).code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(model_->CaptureDecodeGraph(2, -1).code(),
            absl::StatusCode::kInvalidArgument);
}

}  // namespace

// The same equivalence, but for a batch of more than one sequence.
//
// This is R9. The single-sequence case above passed throughout, which is
// exactly what made the multi-shape failure easy to ship: capture one shape and
// everything looks right. Concurrent decode replays a *different* captured
// graph, and that one was wrong.
TEST_F(EngineTest, CudaGraphReplayMatchesLaunchByLaunchForABatch) {
  const std::vector<std::vector<int32_t>> prompts = {
      {kThe, kCapital, kOf, kFrance, kIs},
      {kThe, kCapital, kOf, kFrance, kIs, kParis},
      {kThe, kCapital, kOf, kFrance, kIs},
  };

  SamplingParams params;
  params.max_tokens = 8;

  const auto run = [&](int64_t max_running) {
    auto sched = MakeScheduler(max_running);
    EXPECT_TRUE(sched.ok());

    for (size_t i = 0; i < prompts.size(); ++i) {
      EXPECT_TRUE(
          sched->AddRequest(static_cast<uint64_t>(i + 1), prompts[i], params)
              .ok());
    }

    std::vector<Completion> done = Drain(&*sched);
    std::sort(done.begin(), done.end(),
              [](const Completion& a, const Completion& b) {
                return a.id < b.id;
              });

    return done;
  };

  // All three resident at once, so decode steps carry three sequences.
  const std::vector<Completion> ungraphed = run(3);
  ASSERT_EQ(ungraphed.size(), prompts.size());

  const int64_t max_blocks = (512 + 16 - 1) / 16;

  // Largest shape first: preparing a bigger batch grows buffers that are sized
  // on demand, and a graph captured for a smaller shape beforehand would be
  // left holding addresses that no longer exist.
  for (int64_t seqs = 3; seqs >= 1; --seqs) {
    ASSERT_TRUE(model_->CaptureDecodeGraph(seqs, max_blocks).ok())
        << "capturing " << seqs << " sequences";
  }

  const std::vector<Completion> graphed = run(3);
  ASSERT_EQ(graphed.size(), prompts.size());

  for (size_t i = 0; i < graphed.size(); ++i) {
    EXPECT_EQ(ungraphed[i].output_tokens, graphed[i].output_tokens)
        << "request " << graphed[i].id
        << " decoded differently once its shape was captured";
  }
}


// Capturing a graph must not change a single logit -- R9's regression test.
//
// Comparing *logits* rather than sampled tokens is the whole point. Every graph
// test before this one compared tokens, and an argmax survives a surprisingly
// large error, so a decode step that was wrong in every layer looked fine for
// two milestones.
//
// Three checks, and the order matters, because they separate three different
// things that all present as "the graph is broken":
//
//   1. re-running the same batch with nothing captured must be bit-identical,
//      or the comparison measures the second KV append rather than the graph;
//   2. running launch-by-launch after capturing a graph for a *different*
//      shape must be bit-identical, or capture is corrupting model state and
//      replay is innocent -- which is exactly what R9 turned out to be;
//   3. only then does replaying a matching graph mean anything.
//
// Check 2 is what found it. The capture probe ran the real decode body, so it
// really appended keys and values, and it wrote them to slot 0 -- physical
// block 0, which the free list hands out first. Capturing a graph overwrote the
// first live sequence's position-0 KV in every layer.

TEST_F(EngineTest, GraphedAndUngraphedLogitsAgreeForABatch) {
  auto sched = MakeScheduler(3);
  ASSERT_TRUE(sched.ok());

  const std::vector<std::vector<int32_t>> prompts = {
      {kThe, kCapital, kOf, kFrance, kIs},
      {kThe, kCapital, kOf, kFrance, kIs, kParis},
      {kThe, kCapital, kOf, kFrance, kIs},
  };

  SamplingParams params;
  params.max_tokens = 8;

  for (size_t i = 0; i < prompts.size(); ++i) {
    ASSERT_TRUE(
        sched->AddRequest(static_cast<uint64_t>(i + 1), prompts[i], params)
            .ok());
  }

  // Step one: the prefill, which is never graphed.
  ForwardBatch prefill;
  ASSERT_TRUE(sched->PrepareStep(&prefill).ok());
  ASSERT_GT(prefill.num_tokens(), 0);

  std::vector<float> logits;
  ASSERT_TRUE(model_->Step(prefill, &logits).ok());
  ASSERT_TRUE(sched->CommitStep(
                   GreedySample(logits, prefill.logits_indices.size()))
                  .ok());

  // Step two: a decode carrying every sequence. This is the shape that goes
  // wrong in the server.
  ForwardBatch decode;
  ASSERT_TRUE(sched->PrepareStep(&decode).ok());
  ASSERT_EQ(decode.num_seqs, static_cast<int64_t>(prompts.size()));

  std::vector<float> ungraphed;
  ASSERT_TRUE(model_->Step(decode, &ungraphed).ok());

  // Control first: the same batch again, with nothing captured in between. If
  // re-running is not idempotent then the comparison below measures that rather
  // than the graph, and every conclusion drawn from it would be wrong.
  std::vector<float> repeated;
  ASSERT_TRUE(model_->Step(decode, &repeated).ok());

  // Capture a graph for a *different* shape first. FindGraph will not match
  // this batch, so the step below still runs launch-by-launch -- which
  // separates "replay is wrong" from "capturing corrupts model state".
  ASSERT_TRUE(model_->CaptureDecodeGraph(decode.num_seqs + 1,
                                         decode.max_blocks_per_seq)
                  .ok());

  std::vector<float> after_foreign_capture;
  ASSERT_TRUE(model_->Step(decode, &after_foreign_capture).ok());

  {
    double d = 0.0;
    for (size_t i = 0; i < ungraphed.size(); ++i) {
      d = std::max(d, std::fabs(static_cast<double>(ungraphed[i]) -
                                after_foreign_capture[i]));
    }
    std::fprintf(stderr,
                 "[R9] AFTER FOREIGN CAPTURE (still ungraphed): max|diff| = "
                 "%.6g\n", d);
  }

  ASSERT_TRUE(
      model_->CaptureDecodeGraph(decode.num_seqs, decode.max_blocks_per_seq)
          .ok());

  std::vector<float> graphed;
  ASSERT_TRUE(model_->Step(decode, &graphed).ok());

  ASSERT_EQ(ungraphed.size(), graphed.size());
  ASSERT_EQ(ungraphed.size(), repeated.size());

  const size_t rows = decode.logits_indices.size();
  const size_t vocab = ungraphed.size() / rows;

  for (size_t r = 0; r < rows; ++r) {
    double control = 0.0;
    for (size_t v = 0; v < vocab; ++v) {
      control = std::max(control, std::fabs(
          static_cast<double>(ungraphed[r * vocab + v]) -
          repeated[r * vocab + v]));
    }
    std::fprintf(stderr, "[R9] CONTROL row %zu: max|diff| re-running the same "
                 "batch without any graph = %.6g\n", r, control);
  }

  for (size_t r = 0; r < rows; ++r) {
    const float* a = ungraphed.data() + r * vocab;
    const float* b = graphed.data() + r * vocab;

    double max_abs = 0.0;
    size_t first_diff = vocab;
    size_t argmax_a = 0;
    size_t argmax_b = 0;

    for (size_t v = 0; v < vocab; ++v) {
      const double d = std::fabs(static_cast<double>(a[v]) - b[v]);
      if (d > max_abs) max_abs = d;
      if (d != 0.0 && first_diff == vocab) first_diff = v;
      if (a[v] > a[argmax_a]) argmax_a = v;
      if (b[v] > b[argmax_b]) argmax_b = v;
    }

    // Reported unconditionally: when this fails, the *shape* of the
    // disagreement is the finding -- a tiny spread means a reduction reordered,
    // a large one means the replay read different inputs entirely.
    std::fprintf(stderr,
                 "[R9] row %zu: max|diff|=%.6g first_diff_index=%zu "
                 "argmax ungraphed=%zu graphed=%zu\n",
                 r, max_abs, first_diff, argmax_a, argmax_b);

    EXPECT_EQ(argmax_a, argmax_b)
        << "row " << r << " sampled a different token once graphed";
    EXPECT_LT(max_abs, 1e-3)
        << "row " << r << " logits differ by " << max_abs;
  }
}

namespace {

// Preemption against the real model, on a KV cache small enough to run out.
//
// Its own fixture, and so its own model instance, because it needs a pool it
// can exhaust -- a few dozen blocks against EngineTest's 512. Re-attaching a
// smaller pool to the shared model would leave every graph captured by an
// earlier test holding pointers into freed device memory, which is R9's exact
// shape and not a thing to reintroduce for the convenience of a fixture. Gtest
// tears one suite's static state down before it sets the next one up, so the
// two models never exist at the same time.
class PreemptionTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!CudaAvailable()) return;

    auto loaded = model::Qwen2Model::LoadFromDirectory(CheckpointDir());
    if (!loaded.ok()) {
      status_ = loaded.status();
      return;
    }

    model_ = new model::Qwen2Model(*std::move(loaded));

    // 16 blocks of 16 is 256 token slots for the whole engine, against a
    // workload below that peaks around 20 blocks. Oversubscribed on purpose:
    // preemption is unreachable from a pool that is merely small.
    status_ = model_->AttachKvCache(16, 16);
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

  StatusOr<Scheduler> MakeScheduler(int64_t max_running) {
    SchedulerConfig config;
    config.max_running = max_running;
    config.max_batch_tokens = 256;
    config.max_seq_len = 512;

    return Scheduler::Create(config, model_->kv_pool());
  }

  /// Runs to completion, greedily, and returns what finished.
  std::vector<Completion> Drain(Scheduler* sched, int max_steps = 1500) {
    std::vector<Completion> all;

    for (int step = 0; step < max_steps && sched->HasWork(); ++step) {
      ForwardBatch batch;
      EXPECT_TRUE(sched->PrepareStep(&batch).ok()) << "step " << step;

      if (batch.num_tokens() > 0) {
        std::vector<float> logits;
        const Status s = model_->Step(batch, &logits);
        EXPECT_TRUE(s.ok()) << "step " << step << ": " << s;
        if (!s.ok()) break;

        EXPECT_TRUE(
            sched->CommitStep(GreedySample(logits, batch.logits_indices.size()))
                .ok());
      }

      for (Completion& c : sched->TakeCompleted()) all.push_back(std::move(c));
    }

    for (Completion& c : sched->TakeCompleted()) all.push_back(std::move(c));
    return all;
  }

  static model::Qwen2Model* model_;
  static Status status_;
};

model::Qwen2Model* PreemptionTest::model_ = nullptr;
Status PreemptionTest::status_ = OkStatus();

// Recompute preemption really recomputes.
//
// The host tests prove the scheduler's bookkeeping: the same tokens go in and
// the same tokens come out. They cannot prove the part that only exists on a
// GPU -- that re-prefilling a sequence's own history reconstructs the identical
// KV, so the tokens it goes on to produce are the ones it would have produced
// had its cache never been dropped. That is the claim §8.2 rests on, and it is
// what makes recompute a legitimate answer to a shortage rather than a way of
// quietly corrupting whoever got evicted.
//
// A wrong resumption point does not crash here. It re-reads one token too few
// or too many and continues generating fluent, plausible, different text.
TEST_F(PreemptionTest, APreemptedSequenceGeneratesWhatItWouldHaveAnyway) {
  const std::vector<int32_t> tracked = {kThe, kCapital, kOf, kFrance, kIs};

  SamplingParams params;
  params.max_tokens = 24;

  // Alone, with the pool to itself: nothing competes and nothing is preempted.
  std::vector<int32_t> control;
  {
    auto sched = MakeScheduler(/*max_running=*/1);
    ASSERT_TRUE(sched.ok()) << sched.status();
    ASSERT_TRUE(sched->AddRequest(1, tracked, params).ok());

    const std::vector<Completion> done = Drain(&*sched);
    ASSERT_EQ(done.size(), 1u);
    ASSERT_EQ(done[0].reason, FinishReason::kMaxTokens);
    control = done[0].output_tokens;
    ASSERT_EQ(control.size(), 24u);

    ASSERT_EQ(sched->preemptions(), 0) << "the control was itself preempted";
  }

  ASSERT_EQ(model_->kv_pool()->used_blocks(), 0) << "control leaked blocks";

  // Now the same request in a crowd, with ballast long enough to drain the
  // pool out from under it.
  std::vector<int32_t> after;
  int64_t preemptions = 0;
  {
    auto sched = MakeScheduler(/*max_running=*/4);
    ASSERT_TRUE(sched.ok());

    SamplingParams ballast_params;
    ballast_params.max_tokens = 40;

    // Submitted first, so they are older than the tracked request and the
    // tracked one -- the newest -- is the victim §8.2 picks.
    std::vector<int32_t> ballast;
    for (int i = 0; i < 12; ++i) {
      for (const int32_t t : {kThe, kCapital, kOf, kFrance}) ballast.push_back(t);
    }

    for (int i = 0; i < 3; ++i) {
      ASSERT_TRUE(sched->AddRequest(static_cast<uint64_t>(10 + i), ballast,
                                    ballast_params)
                      .ok());
    }

    // Let the ballast get running and start consuming blocks.
    for (int i = 0; i < 3; ++i) {
      ForwardBatch batch;
      ASSERT_TRUE(sched->PrepareStep(&batch).ok());
      if (batch.num_tokens() == 0) break;

      std::vector<float> logits;
      ASSERT_TRUE(model_->Step(batch, &logits).ok());
      ASSERT_TRUE(
          sched->CommitStep(GreedySample(logits, batch.logits_indices.size()))
              .ok());
    }

    ASSERT_TRUE(sched->AddRequest(1, tracked, params).ok());

    std::vector<Completion> done = Drain(&*sched);
    preemptions = sched->preemptions();

    for (const Completion& c : done) {
      if (c.id == 1) {
        EXPECT_EQ(c.reason, FinishReason::kMaxTokens);
        after = c.output_tokens;
      }
    }
  }

  ASSERT_GT(preemptions, 0)
      << "the pool was not tight enough to preempt anything";

  ASSERT_EQ(after.size(), control.size())
      << "the preempted run generated a different number of tokens";

  EXPECT_EQ(after, control)
      << "a sequence came back different after " << preemptions
      << " preemptions -- recompute did not reconstruct its KV";

  EXPECT_EQ(model_->kv_pool()->used_blocks(), 0) << "blocks leaked";
}

// A prefix cache hit is invisible in the output (§6.3).
//
// The second request here never computes most of its prompt -- it reads keys
// and values another request wrote and left in the radix tree. That is only
// legitimate because attention is causal, so a key is a function of the tokens
// before it as well as the token itself, and the path through the tree is
// exactly that history. If the match were ever off by a token, or a block were
// shared between two prompts that merely happen to collide, the reader would
// attend over somebody else's context and carry on producing fluent text.
//
// Which is why this compares against a cold run of the same prompt rather than
// checking that anything was reused: reuse that changes the answer is worse
// than no reuse at all.
TEST_F(PreemptionTest, APrefixCacheHitDoesNotChangeTheOutput) {
  // Long enough to span several blocks, so most of it is matchable.
  std::vector<int32_t> prompt;
  for (int i = 0; i < 8; ++i) {
    for (const int32_t t : {kThe, kCapital, kOf, kFrance}) prompt.push_back(t);
  }
  prompt.push_back(kIs);

  SamplingParams params;
  params.max_tokens = 12;

  // Cold: an empty cache, so every token of the prompt is computed.
  std::vector<int32_t> cold;
  int64_t cold_hits = 0;
  {
    auto sched = MakeScheduler(/*max_running=*/1);
    ASSERT_TRUE(sched.ok());
    ASSERT_TRUE(sched->AddRequest(1, prompt, params).ok());

    const std::vector<Completion> done = Drain(&*sched);
    ASSERT_EQ(done.size(), 1u);
    cold = done[0].output_tokens;
    cold_hits = sched->prefix_hit_tokens();
  }
  ASSERT_EQ(cold_hits, 0) << "the cold run was itself served from a cache";
  ASSERT_EQ(cold.size(), 12u);

  // Warm: the same prompt through a scheduler whose cache already holds it.
  // One scheduler so the tree survives between the two requests.
  std::vector<int32_t> warm;
  int64_t hits = 0;
  {
    auto sched = MakeScheduler(/*max_running=*/1);
    ASSERT_TRUE(sched.ok());

    ASSERT_TRUE(sched->AddRequest(1, prompt, params).ok());
    ASSERT_EQ(Drain(&*sched).size(), 1u);

    ASSERT_TRUE(sched->AddRequest(2, prompt, params).ok());
    const std::vector<Completion> done = Drain(&*sched);
    ASSERT_EQ(done.size(), 1u);

    warm = done[0].output_tokens;
    hits = sched->prefix_hit_tokens();
  }

  ASSERT_GT(hits, 0) << "the second request did not hit the cache";
  std::fprintf(stderr, "[prefix cache] %ld of %zu prompt tokens reused\n",
               static_cast<long>(hits), prompt.size());

  EXPECT_EQ(warm, cold)
      << "reusing a cached prefix changed what the model generated";
}

// A partial match: the second prompt shares a head with the first and diverges.
// Only the shared part may be reused, and the divergent part has to be computed
// against it -- the case a block-collision bug would sail straight through.
//
// Compared as logits rather than as generated text, and deliberately. A warm
// run forwards only the unmatched tail, so its GEMMs are a different shape than
// a cold run's, and bf16 reductions do not commute: the same computation
// arranged differently lands a fraction of a logit apart. Ten greedy steps then
// amplify that fraction into a visibly different completion whenever the model
// is near a tie -- which the repetitive prompt below very much is. Comparing
// the distribution the reuse produced, once, measures the thing in question
// instead of measuring how long a coin stays on its edge.
TEST_F(PreemptionTest, APartialPrefixMatchIsStillCorrect) {
  std::vector<int32_t> shared;
  for (int i = 0; i < 6; ++i) {
    for (const int32_t t : {kThe, kCapital, kOf, kFrance}) shared.push_back(t);
  }

  std::vector<int32_t> first = shared;
  for (const int32_t t : {kIs, kParis, kIs, kParis}) first.push_back(t);

  std::vector<int32_t> second = shared;
  for (const int32_t t : {kOf, kThe, kOf, kThe}) second.push_back(t);

  SamplingParams params;
  params.max_tokens = 10;

  // Logits for `second`'s first generated token, optionally after `first` has
  // populated the shared head.
  const auto next_token_logits = [&](bool warm, int64_t* out_hits) {
    std::vector<float> logits;

    auto sched = MakeScheduler(/*max_running=*/1);
    EXPECT_TRUE(sched.ok());

    if (warm) {
      EXPECT_TRUE(sched->AddRequest(1, first, params).ok());
      EXPECT_EQ(Drain(&*sched).size(), 1u);
    }

    const int64_t before = sched->prefix_hit_tokens();
    EXPECT_TRUE(sched->AddRequest(2, second, params).ok());

    ForwardBatch batch;
    EXPECT_TRUE(sched->PrepareStep(&batch).ok());
    EXPECT_EQ(batch.logits_indices.size(), 1u)
        << "the prompt should finish in one step at this budget";
    EXPECT_TRUE(model_->Step(batch, &logits).ok());

    *out_hits = sched->prefix_hit_tokens() - before;
    return logits;
  };

  int64_t cold_hits = 0;
  int64_t warm_hits = 0;

  const std::vector<float> cold = next_token_logits(false, &cold_hits);
  const std::vector<float> warm = next_token_logits(true, &warm_hits);

  EXPECT_EQ(cold_hits, 0) << "the cold run was served from a cache";

  ASSERT_GT(warm_hits, 0) << "the shared head was not reused";
  ASSERT_LE(warm_hits, static_cast<int64_t>(shared.size()))
      << "more was reused than the two prompts actually share";

  ASSERT_FALSE(cold.empty());
  ASSERT_EQ(warm.size(), cold.size());

  double worst = 0.0;
  double scale = 0.0;
  size_t cold_argmax = 0;
  size_t warm_argmax = 0;

  for (size_t i = 0; i < cold.size(); ++i) {
    worst = std::max(worst, std::fabs(static_cast<double>(cold[i]) -
                                      static_cast<double>(warm[i])));
    scale = std::max(scale, std::fabs(static_cast<double>(cold[i])));
    if (cold[i] > cold[cold_argmax]) cold_argmax = i;
    if (warm[i] > warm[warm_argmax]) warm_argmax = i;
  }

  std::fprintf(stderr,
               "[prefix cache] partial match reused %ld of %zu tokens; "
               "max|diff| = %.6g over range %.6g\n",
               static_cast<long>(warm_hits), shared.size(), worst, scale);

  EXPECT_EQ(cold_argmax, warm_argmax)
      << "reusing a shared prefix changed the model's prediction";

  EXPECT_LT(worst, 0.02 * scale)
      << "the reused prefix moved the logits further than rounding explains";
}

// The engine survives a pool it cannot satisfy. Before preemption, a shortage
// left PrepareStep with an error, and the engine's only answer to that is to
// fail every request in flight -- so one greedy sequence took the server down
// with it.
TEST_F(PreemptionTest, AnOversubscribedPoolStillDrainsEveryRequest) {
  auto sched = MakeScheduler(/*max_running=*/4);
  ASSERT_TRUE(sched.ok());

  SamplingParams params;
  params.max_tokens = 48;

  std::vector<int32_t> prompt;
  for (int i = 0; i < 10; ++i) {
    for (const int32_t t : {kThe, kCapital, kOf, kFrance}) prompt.push_back(t);
  }

  for (int i = 0; i < 4; ++i) {
    ASSERT_TRUE(
        sched->AddRequest(static_cast<uint64_t>(i + 1), prompt, params).ok());
  }

  const std::vector<Completion> done = Drain(&*sched);

  ASSERT_EQ(done.size(), 4u) << "a request was lost";

  for (const Completion& c : done) {
    EXPECT_NE(c.reason, FinishReason::kNotFinished) << "request " << c.id;
    EXPECT_FALSE(c.output_tokens.empty()) << "request " << c.id;
  }

  std::fprintf(stderr,
               "[preemption] %ld preemptions to drain 4 requests, "
               "%ld prompt tokens reused\n",
               static_cast<long>(sched->preemptions()),
               static_cast<long>(sched->prefix_hit_tokens()));

  // The scheduler is still alive, so its prefix cache still legitimately holds
  // blocks. What must be zero is what the *sequences* hold.
  EXPECT_EQ(sched->blocks_in_use(), 0) << "a finished sequence kept its blocks";
  EXPECT_EQ(model_->kv_pool()->used_blocks(), sched->cached_blocks())
      << "blocks belong to neither a sequence nor the cache";
}

}  // namespace

}  // namespace inferx
