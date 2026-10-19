// Preemption tests: KV pressure preempts the newest running request, the
// victim recomputes (from its own cached prefix where caching is on), and
// cached blocks survive preemption (vLLM V1 recompute semantics).

#include "inferx/engine/scheduler.h"

#include "inferx/ops/attention/registry.h"
#include "inferx/ops/attention/runtime.h"
#include "inferx/ops/attention/spec.h"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "inferx/core/dtype.h"

namespace inferx {
namespace {

constexpr TokenId kEos = 99;
constexpr TokenId kTok = 7;
constexpr int64_t kBlockSize = 4;

/// Two dense GQA layers over one CPU pool, the shape every scheduler test
/// schedules against (matches the legacy 2-layer KvLayout pools).
attn::AttentionSpec LayerSpec() {
  attn::AttentionSpec spec;
  spec.params = attn::DenseParams{.num_q_heads = 4, .num_kv_heads = 1, .head_dim = 8};
  spec.q_dtype = DataType::kBFloat16;
  spec.kv_dtype = DataType::kBFloat16;
  return spec;
}

std::unique_ptr<attn::AttentionRuntime> MakeRuntime(bool enable_prefix_caching,
                                                    int64_t blocks) {
  auto registry = std::make_unique<attn::BackendRegistry>();
  attn::RegisterBuiltinBackends(*registry);
  attn::AttentionConfig config;
  config.block_size = kBlockSize;
  config.max_model_len = 4096;
  config.max_seqs = 256;
  config.max_batch_tokens = 4096;
  // 2 layers x 2*1*8*2 bytes/token x 4 tokens/block = 256 bytes/block.
  config.cache_bytes_budget = blocks * 256;
  config.device = Device::Cpu();
  config.enable_prefix_caching = enable_prefix_caching;
  config.forced_backend = "reference";
  const std::vector<attn::AttentionSpec> specs{LayerSpec(), LayerSpec()};
  return std::move(
      attn::AttentionRuntime::Create(std::move(registry), config, specs).value());
}


class SchedulerPreemptTest : public ::testing::Test {
 protected:
  void MakeScheduler(bool enable_prefix_caching, int64_t blocks) {
    runtime_ = MakeRuntime(enable_prefix_caching, blocks);
    scheduler_ = std::make_unique<Scheduler>(SchedulerConfig{}, std::vector<TokenId>{kEos},
                                             *runtime_);
  }

  bool Step() {
    auto plan = scheduler_->Schedule();
    EXPECT_TRUE(plan.ok()) << plan.status();
    if (!plan.ok()) return false;
    last_plan_ = std::move(*plan);
    ModelRunnerOutput result;
    for (const ScheduledRequest& sr : last_plan_.scheduled) {
      result.samples.push_back({sr.request_id, {kTok}});
    }
    EXPECT_TRUE(scheduler_->UpdateFromOutput(last_plan_, result).ok());
    return true;
  }

  sampling::SamplingParams LongParams(int tokens) {
    sampling::SamplingParams params;
    params.max_tokens = tokens;
    params.ignore_eos = true;
    return params;
  }

  std::vector<TokenId> Tokens(int64_t count, int64_t base = 100) {
    std::vector<TokenId> tokens;
    tokens.reserve(count);
    for (int64_t i = 0; i < count; ++i) tokens.push_back(static_cast<TokenId>(base + i));
    return tokens;
  }

  std::unique_ptr<attn::AttentionRuntime> runtime_;
  std::unique_ptr<Scheduler> scheduler_;
  SchedulerOutput last_plan_;
};

TEST_F(SchedulerPreemptTest, PressurePreemptsNewestAndVictimResumes) {
  // Six blocks (24 tokens): each request needs five alone (8 prompt + 12
  // generated = 20 tokens), so they overflow only when running together.
  MakeScheduler(false, /*blocks=*/6);
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, Tokens(8), LongParams(12))).ok());
  ASSERT_TRUE(Step());
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, Tokens(8), LongParams(12))).ok());
  ASSERT_TRUE(Step());
  // Both decodes now grow one token per step until the pool runs dry.
  for (int i = 0; i < 100 && scheduler_->HasRequests(); ++i) ASSERT_TRUE(Step());

  // The pool is exhausted: whichever request hits the wall first preempts
  // the newer one, finishes alone, and the victim then completes. Drain
  // everything and check no request was lost.
  for (int i = 0; i < 500 && scheduler_->HasRequests(); ++i) ASSERT_TRUE(Step());
  EXPECT_FALSE(scheduler_->HasRequests());
  int finished = 0;
  while (scheduler_->PopFinished().has_value()) ++finished;
  EXPECT_EQ(finished, 2);
  EXPECT_GE(scheduler_->Stats().num_preempted, 1);
  EXPECT_EQ(runtime_->BlockPool()->FreeBlocks(), runtime_->BlockPool()->NumBlocks());  // No leaks.
}

TEST_F(SchedulerPreemptTest, PreemptedRequestReadoptsItsOwnCachedPrefix) {
  MakeScheduler(true, /*blocks=*/6);
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, Tokens(8), LongParams(12))).ok());
  ASSERT_TRUE(Step());  // Prompt cached (two full blocks).

  ASSERT_TRUE(scheduler_->AddRequest(Request(2, Tokens(8), LongParams(12))).ok());
  ASSERT_TRUE(Step());

  // Grow both until request 2 (the newer) is preempted.
  // The victim can return in the very step that preempted it: preemption
  // happens in the running pass and admission follows in the same step.
  // Resumption replays prompt plus retained output (C05) and re-adopts the
  // cached prefix of that sequence: both prompt blocks were published and
  // stay pinned, so at least eight tokens come from cache; more when the
  // victim's own output blocks had completed before preemption.
  bool saw_preemption = false;
  bool resumed = false;
  for (int i = 0; i < 500 && scheduler_->HasRequests(); ++i) {
    ASSERT_TRUE(Step());
    if (!last_plan_.preempted_request_ids.empty()) {
      ASSERT_EQ(last_plan_.preempted_request_ids,
                (std::vector<RequestId>{2}));  // Newest loses.
      saw_preemption = true;
    }
    for (const NewRequestData& fresh : last_plan_.scheduled_new_reqs) {
      if (fresh.request_id != 2) continue;
      ASSERT_TRUE(saw_preemption);
      resumed = true;
      EXPECT_GE(fresh.num_computed_tokens, 8);  // Cached prompt blocks back.
    }
    if (saw_preemption && resumed) break;
  }
  ASSERT_TRUE(saw_preemption);
  EXPECT_TRUE(resumed);
  for (int i = 0; i < 500 && scheduler_->HasRequests(); ++i) ASSERT_TRUE(Step());
}

TEST_F(SchedulerPreemptTest, PreemptedVictimReplaysAcceptedOutput) {
  // C05: a preempted victim keeps its accepted output and the readmission
  // plan hands the runner the whole sequence -- prompt plus every retained
  // output token -- so replay recomputes KV for exactly the tokens the
  // continuation attends over, and no old token is ever re-emitted.
  MakeScheduler(false, /*blocks=*/6);
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, Tokens(8), LongParams(12))).ok());
  ASSERT_TRUE(Step());
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, Tokens(8), LongParams(16))).ok());
  ASSERT_TRUE(Step());

  bool saw_preemption = false;
  bool saw_replay = false;
  for (int i = 0; i < 500 && scheduler_->HasRequests(); ++i) {
    ASSERT_TRUE(Step());
    if (!saw_preemption && !last_plan_.preempted_request_ids.empty()) {
      ASSERT_EQ(last_plan_.preempted_request_ids, (std::vector<RequestId>{2}));
      saw_preemption = true;
    }
    for (const NewRequestData& fresh : last_plan_.scheduled_new_reqs) {
      if (fresh.request_id != 2) continue;
      ASSERT_TRUE(saw_preemption);  // Readmission follows preemption.
      if (saw_replay) continue;     // ... and happens once.
      saw_replay = true;
      // Prompt first (Tokens(8) = 100..107), then retained output (kTok).
      ASSERT_GT(fresh.prompt_token_ids.size(), 8u);
      for (int t = 0; t < 8; ++t) {
        EXPECT_EQ(fresh.prompt_token_ids[t], static_cast<TokenId>(100 + t));
      }
      for (size_t t = 8; t < fresh.prompt_token_ids.size(); ++t) {
        EXPECT_EQ(fresh.prompt_token_ids[t], kTok);
      }
      EXPECT_EQ(fresh.num_computed_tokens, 0);  // Recompute from zero.
    }
  }
  ASSERT_TRUE(saw_preemption);
  ASSERT_TRUE(saw_replay);
  for (int i = 0; i < 500 && scheduler_->HasRequests(); ++i) ASSERT_TRUE(Step());

  // Both complete; the victim's output continues past its preemption point
  // rather than restarting, and reports the preemption it suffered.
  int finished = 0;
  std::optional<Request> victim;
  while (auto done = scheduler_->PopFinished()) {
    ++finished;
    if (done->GetRequestId() == 2) victim = std::move(*done);
  }
  ASSERT_EQ(finished, 2);
  ASSERT_TRUE(victim.has_value());
  EXPECT_EQ(victim->NumOutputTokens(), 16);
  EXPECT_EQ(victim->NumTokens(), 8 + 16);
  EXPECT_GE(victim->NumPreemptions(), 1);
  EXPECT_EQ(runtime_->BlockPool()->FreeBlocks(), runtime_->BlockPool()->NumBlocks());  // No leaks.
}

TEST_F(SchedulerPreemptTest, RunnerStateWouldDropPreemptedRequests) {
  // The output contract: preempted ids ride along so the runner can forget
  // them (they return as fresh requests later).
  MakeScheduler(false, /*blocks=*/5);
  ASSERT_TRUE(scheduler_->AddRequest(Request(1, Tokens(12), LongParams(4))).ok());
  ASSERT_TRUE(scheduler_->AddRequest(Request(2, Tokens(8), LongParams(4))).ok());

  bool saw = false;
  for (int i = 0; i < 500 && scheduler_->HasRequests(); ++i) {
    ASSERT_TRUE(Step());
    if (!last_plan_.preempted_request_ids.empty()) {
      saw = true;
      // A step never schedules a request it preempts.
      for (const ScheduledRequest& sr : last_plan_.scheduled) {
        EXPECT_NE(sr.request_id, last_plan_.preempted_request_ids.front());
      }
      break;
    }
  }
  EXPECT_TRUE(saw);
}

}  // namespace
}  // namespace inferx
