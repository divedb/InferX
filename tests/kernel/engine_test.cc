// The engine, end to end: scheduler + paged cache + real model.
//
// M3's deliverable is "the engine exists", and this is what that means -- a
// request goes in, a step loop runs, tokens come out, blocks are returned. The
// scheduler contributes no CUDA and the model contributes no policy; the loop
// below is the entire seam between them, and it is §5.2's depth-0 case:
// prepare, execute, commit, nothing in flight.

#include <algorithm>
#include <cmath>
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

  StatusOr<Scheduler> MakeScheduler(int64_t max_running = 4) {
    SchedulerConfig config;
    config.max_running = max_running;
    config.max_batch_tokens = 2048;
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

  EXPECT_EQ(model_->kv_pool()->used_blocks(), 0);
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

}  // namespace inferx
