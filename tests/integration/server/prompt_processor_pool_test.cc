// Integration tests for the bounded prompt-processor pool
// (docs/tokenizer_design.md §8.2): token-for-token parity with the facade,
// pre-tokenized input validation, bounded admission, cancellation
// before/during prepare, request timeouts, and cooperative drain.
// Determinism comes from holding a worker with a slow encode (a large
// many-word text) and polling Health for observed queue/busy state before
// issuing the action under test.
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <gtest/gtest.h>

#include "inferx/core/status.h"
#include "inferx/server/prompt_processor.h"
#include "inferx/server/prompt_processor_pool.h"
#include "inferx/server/request_model.h"
#include "inferx/tokenizer/tokenizer.h"

namespace net = boost::asio;
using inferx::Status;
using inferx::StatusOr;
using inferx::Tokenizer;
using inferx::PoolHealth;
using inferx::PrepareContext;
using inferx::PreparedPrompt;
using inferx::PromptInput;
using inferx::PromptPoolConfig;
using inferx::PromptProcessorPool;
using inferx::TextPrompt;
using inferx::TokenIdsPrompt;
using PoolClock = std::chrono::steady_clock;

namespace {

constexpr const char* kTokenizerPath =
    "tests/testdata/tiny_tokenizer/tokenizer.json";

PromptPoolConfig BaseConfig() {
  PromptPoolConfig config;
  config.workers = 1;
  config.tokenizer_path = kTokenizerPath;
  // The slow-encode jobs are 8 MiB texts; keep admission focused on the
  // request/byte budgets the individual tests exercise.
  config.max_prompt_bytes = 64u << 20;
  config.queue_max_bytes = 64u << 20;
  return config;
}

/// A many-word text of roughly `bytes`: the whitespace pre-tokenizer keeps
/// encoding linear while total volume makes an encode last long enough to
/// hold a worker deterministically.
std::string SlowText(std::size_t bytes) {
  std::string text;
  text.reserve(bytes + 64);
  while (text.size() < bytes) text += "aaaaaaaabbbbcccc aaaaabbbbcc ";
  return text;
}

/// Coroutine wrappers as functors: lambdas cannot be coroutines portably.
struct PrepareTask {
  PrepareTask(PromptProcessorPool& pool, std::uint64_t id, PromptInput input)
      : pool(pool), id(id), input(std::move(input)) {}

  PromptProcessorPool& pool;
  std::uint64_t id;
  PromptInput input;
  PrepareContext context;
  net::awaitable<StatusOr<PreparedPrompt>> operator()() {
    context.request_id = id;
    co_return co_await pool.Prepare(id, std::move(input), context);
  }
};

StatusOr<PreparedPrompt> PrepareSync(net::io_context& io, PromptProcessorPool& pool,
                                     std::uint64_t id, PromptInput input) {
  auto future = net::co_spawn(io, PrepareTask{pool, id, std::move(input)},
                              net::use_future);
  return future.get();
}

/// Polls `predicate` on the test thread until it holds or `timeout` passes.
template <typename Predicate>
bool WaitFor(Predicate predicate, std::chrono::milliseconds timeout) {
  const auto deadline = PoolClock::now() + timeout;
  while (PoolClock::now() < deadline) {
    if (predicate()) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
  }
  return predicate();
}

/// A pool plus the io thread servicing its coroutines and deadline sweep.
class PoolHarness {
 public:
  explicit PoolHarness(PromptPoolConfig config)
      : pool_(inferx::CreatePromptProcessorPool(io_, std::move(config))) {
    io_thread_ = std::thread([this] { io_.run(); });
  }
  ~PoolHarness() {
    io_.stop();
    io_thread_.join();
  }

  PromptProcessorPool& pool() { return *pool_; }
  net::io_context& io() { return io_; }
  Status StartAndWaitReady() {
    return pool_->WaitUntilReady(std::chrono::milliseconds(30000));
  }
  PoolHealth Health() const { return pool_->Health(); }

 private:
  net::io_context io_;
  std::shared_ptr<PromptProcessorPool> pool_;
  std::thread io_thread_;
};

}  // namespace

TEST(PromptProcessorPoolTest, StartupGateFailsForBadTokenizerPath) {
  PromptPoolConfig config = BaseConfig();
  config.tokenizer_path = "tests/testdata/tiny_tokenizer/missing.json";
  PoolHarness harness(config);
  const Status ready = harness.StartAndWaitReady();
  ASSERT_FALSE(ready.ok());
  EXPECT_EQ(ready.code(), absl::StatusCode::kFailedPrecondition);
}

TEST(PromptProcessorPoolTest, PrepareMatchesFacadeTokenForToken) {
  auto facade = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(facade.ok()) << facade.status();

  PoolHarness harness(BaseConfig());
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  const std::vector<std::string> inputs = {
      "abc", "a", "abab", "hello world", "héllo 世界", std::string(4096, 'a')};
  std::uint64_t id = 1;
  for (const std::string& input : inputs) {
    const auto expected = (*facade)->Encode(input);
    ASSERT_TRUE(expected.ok()) << expected.status();
    const auto prepared =
        PrepareSync(harness.io(), harness.pool(), id, PromptInput{TextPrompt{input}});
    ASSERT_TRUE(prepared.ok()) << prepared.status();
    EXPECT_EQ(prepared->token_ids, *expected) << "input: " << input;
    EXPECT_EQ(prepared->prompt_tokens, static_cast<int>(expected->size()));
    ASSERT_NE(prepared->tokenizer_fingerprint, 0u);
    ++id;
  }
}

TEST(PromptProcessorPoolTest, TokenIdsInputPassesThrough) {
  auto facade = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(facade.ok()) << facade.status();
  const auto ids = (*facade)->Encode("hello world");
  ASSERT_TRUE(ids.ok()) << ids.status();

  PoolHarness harness(BaseConfig());
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  const auto prepared = PrepareSync(harness.io(), harness.pool(), 1,
                                    PromptInput{TokenIdsPrompt{*ids}});
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  EXPECT_EQ(prepared->token_ids, *ids);

  const auto negative = PrepareSync(harness.io(), harness.pool(), 2,
                                    PromptInput{TokenIdsPrompt{{-1, 5}}});
  ASSERT_FALSE(negative.ok());
  EXPECT_EQ(negative.status().code(), absl::StatusCode::kInvalidArgument);

  const auto overflow = PrepareSync(
      harness.io(), harness.pool(), 3,
      PromptInput{TokenIdsPrompt{{static_cast<int>((*facade)->vocab_size())}}});
  ASSERT_FALSE(overflow.ok());
  EXPECT_EQ(overflow.status().code(), absl::StatusCode::kInvalidArgument);

  const auto empty = PrepareSync(harness.io(), harness.pool(), 4,
                                 PromptInput{TokenIdsPrompt{{}}});
  ASSERT_FALSE(empty.ok());
  EXPECT_EQ(empty.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(PromptProcessorPoolTest, ChatAndMultimodalAreExplicitlyUnimplemented) {
  PoolHarness harness(BaseConfig());
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  const auto chat = PrepareSync(harness.io(), harness.pool(), 1,
                                PromptInput{inferx::ChatPrompt{}});
  ASSERT_FALSE(chat.ok());
  EXPECT_EQ(chat.status().code(), absl::StatusCode::kUnimplemented);

  const auto multimodal = PrepareSync(harness.io(), harness.pool(), 2,
                                      PromptInput{inferx::MultimodalInput{}});
  ASSERT_FALSE(multimodal.ok());
  EXPECT_EQ(multimodal.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(PromptProcessorPoolTest, OversizedPromptRejectedWithoutWorker) {
  PromptPoolConfig config = BaseConfig();
  config.queue_max_bytes = 1024;
  PoolHarness harness(config);
  const auto result =
      PrepareSync(harness.io(), harness.pool(), 1, PromptInput{TextPrompt{SlowText(4096)}});
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(harness.Health().metrics.rejected_oversize, 1u);
}

TEST(PromptProcessorPoolTest, PromptTokenLimitRejectsAfterEncode) {
  PoolHarness harness(BaseConfig());
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  // A limit of one token rejects everything longer after encoding.
  PrepareTask task{harness.pool(), 1, PromptInput{TextPrompt{"hello world"}}};
  task.context.limits.max_prompt_tokens = 1;
  const auto result =
      net::co_spawn(harness.io(), std::move(task), net::use_future).get();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(PromptProcessorPoolTest, AdmissionBoundRejectsAndThenServes) {
  auto facade = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(facade.ok()) << facade.status();

  PromptPoolConfig config = BaseConfig();
  config.queue_max_requests = 1;
  config.queue_max_bytes = 64u << 20;
  PoolHarness harness(config);
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  const std::string text = SlowText(8 << 20);
  auto first = net::co_spawn(harness.io(),
                             PrepareTask{harness.pool(), 1, PromptInput{TextPrompt{text}}},
                             net::use_future);
  ASSERT_TRUE(WaitFor([&] { return harness.Health().workers_busy == 1; },
                      std::chrono::seconds(10)))
      << "worker never picked the job up";

  // One running job fills the request budget: the next Prepare is bounded
  // out (the HTTP layer's 503 path) without waiting on the worker.
  const auto second =
      PrepareSync(harness.io(), harness.pool(), 2, PromptInput{TextPrompt{"ab"}});
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(harness.Health().metrics.rejected_full, 1u);

  const auto expected = (*facade)->Encode(text);
  ASSERT_TRUE(expected.ok()) << expected.status();
  const auto prepared = first.get();
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  EXPECT_EQ(prepared->token_ids, *expected);
}

TEST(PromptProcessorPoolTest, CancelQueuedRequestCompletesCancelled) {
  PromptPoolConfig config = BaseConfig();
  config.queue_max_requests = 8;
  config.queue_max_bytes = 64u << 20;
  PoolHarness harness(config);
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  auto first = net::co_spawn(
      harness.io(), PrepareTask{harness.pool(), 1, PromptInput{TextPrompt{SlowText(8 << 20)}}},
      net::use_future);
  ASSERT_TRUE(WaitFor([&] { return harness.Health().workers_busy == 1; },
                      std::chrono::seconds(10)));
  auto second = net::co_spawn(harness.io(),
                              PrepareTask{harness.pool(), 2, PromptInput{TextPrompt{"ab"}}},
                              net::use_future);
  ASSERT_TRUE(WaitFor([&] { return harness.Health().queued_requests == 1; },
                      std::chrono::seconds(10)));

  harness.pool().Cancel(2);
  const auto cancelled = second.get();
  ASSERT_FALSE(cancelled.ok());
  EXPECT_EQ(cancelled.status().code(), absl::StatusCode::kCancelled);

  const auto prepared = first.get();
  ASSERT_TRUE(prepared.ok()) << prepared.status();
}

TEST(PromptProcessorPoolTest, CancelInFlightKeepsWorkerUsable) {
  PoolHarness harness(BaseConfig());
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  auto job = net::co_spawn(
      harness.io(), PrepareTask{harness.pool(), 7, PromptInput{TextPrompt{SlowText(8 << 20)}}},
      net::use_future);
  ASSERT_TRUE(WaitFor([&] { return harness.Health().workers_busy == 1; },
                      std::chrono::seconds(10)));

  harness.pool().Cancel(7);
  const auto cancelled = job.get();
  ASSERT_FALSE(cancelled.ok());
  EXPECT_EQ(cancelled.status().code(), absl::StatusCode::kCancelled);

  // The late result must be discarded and the worker must keep serving.
  const auto next =
      PrepareSync(harness.io(), harness.pool(), 8, PromptInput{TextPrompt{"ab"}});
  ASSERT_TRUE(next.ok()) << next.status();
}

TEST(PromptProcessorPoolTest, SlowEncodeTimesOutWithDeadlineExceeded) {
  PromptPoolConfig config = BaseConfig();
  config.request_timeout = std::chrono::milliseconds(200);
  config.queue_max_bytes = 64u << 20;
  PoolHarness harness(config);
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  auto job = net::co_spawn(
      harness.io(), PrepareTask{harness.pool(), 1, PromptInput{TextPrompt{SlowText(8 << 20)}}},
      net::use_future);
  const auto timed_out = job.get();
  ASSERT_FALSE(timed_out.ok());
  EXPECT_EQ(timed_out.status().code(), absl::StatusCode::kDeadlineExceeded);
  EXPECT_EQ(harness.Health().metrics.timed_out, 1u);

  // The abandoned encode still finishes; accounting recovers and the pool
  // keeps serving afterwards.
  ASSERT_TRUE(WaitFor([&] { return harness.Health().queued_bytes == 0; },
                      std::chrono::seconds(10)));
  const auto next =
      PrepareSync(harness.io(), harness.pool(), 2, PromptInput{TextPrompt{"ab"}});
  ASSERT_TRUE(next.ok()) << next.status();
}

TEST(PromptProcessorPoolTest, DrainRejectsNewWorkAndJoinsCleanly) {
  PromptPoolConfig config = BaseConfig();
  config.workers = 2;
  config.queue_max_bytes = 64u << 20;
  config.shutdown_grace = std::chrono::milliseconds(3000);
  PoolHarness harness(config);
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  // Slow jobs keep both workers observably busy so drain sees them admitted.
  auto first = net::co_spawn(
      harness.io(), PrepareTask{harness.pool(), 1, PromptInput{TextPrompt{SlowText(8 << 20)}}},
      net::use_future);
  auto second = net::co_spawn(
      harness.io(), PrepareTask{harness.pool(), 2, PromptInput{TextPrompt{SlowText(8 << 20)}}},
      net::use_future);
  ASSERT_TRUE(WaitFor([&] { return harness.Health().workers_busy == 2; },
                      std::chrono::seconds(10)));

  harness.pool().BeginDrain();
  const auto rejected =
      PrepareSync(harness.io(), harness.pool(), 3, PromptInput{TextPrompt{"a"}});
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);

  // Admitted work drains through the workers rather than being cancelled.
  const auto a = first.get();
  const auto b = second.get();
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();

  struct JoinTask {
    PromptProcessorPool& pool;
    PoolClock::time_point deadline;
    net::awaitable<Status> operator()() { co_return co_await pool.JoinUntil(deadline); }
  };
  auto joined = net::co_spawn(
      harness.io(),
      JoinTask{harness.pool(), PoolClock::now() + std::chrono::seconds(5)},
      net::use_future);
  EXPECT_TRUE(joined.get().ok());
  const PoolHealth health = harness.Health();
  EXPECT_TRUE(health.draining);
  EXPECT_EQ(health.queued_requests, 0u);
}

TEST(PromptProcessorPoolTest, ConfigValidationRejectsBadValues) {
  PromptPoolConfig config = BaseConfig();
  ASSERT_TRUE(config.Validate().ok()) << config.Validate();

  PromptPoolConfig empty_path = BaseConfig();
  empty_path.tokenizer_path.clear();
  EXPECT_FALSE(empty_path.Validate().ok());

  PromptPoolConfig zero_workers = BaseConfig();
  zero_workers.workers = 0;
  EXPECT_FALSE(zero_workers.Validate().ok());

  PromptPoolConfig oversize_prompt = BaseConfig();
  oversize_prompt.max_prompt_bytes = oversize_prompt.queue_max_bytes + 1;
  EXPECT_FALSE(oversize_prompt.Validate().ok());
}
