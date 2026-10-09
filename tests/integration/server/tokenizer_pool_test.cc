// Integration tests for the thread-executor TokenizerPool: token-for-token
// parity with the in-process facade, bounded admission, cancellation
// before/during encode, request timeouts, and cooperative drain.
// Determinism comes from holding a worker with a slow encode (a large
// many-word text) and polling Health for observed queue/busy state before
// issuing the action under test.
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <gtest/gtest.h>

#include "inferx/server/tokenizer_pool.h"
#include "inferx/tokenizer/tokenizer.h"

namespace net = boost::asio;
using inferx::server::PoolHealth;
using inferx::server::PreparedPrompt;
using inferx::server::TokenizerPool;
using inferx::server::TokenizerPoolConfig;
using inferx::Status;
using inferx::StatusOr;
using PoolClock = std::chrono::steady_clock;

namespace {

constexpr const char* kTokenizerPath =
    "tests/testdata/tiny_tokenizer/tokenizer.json";

TokenizerPoolConfig BaseConfig() {
  TokenizerPoolConfig config;
  config.workers = 1;
  config.tokenizer_path = kTokenizerPath;
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

/// Co-routine wrappers as functors: lambdas cannot be coroutines portably.
struct PrepareTask {
  TokenizerPool& pool;
  std::uint64_t id;
  std::string text;
  net::awaitable<StatusOr<PreparedPrompt>> operator()() {
    co_return co_await pool.Prepare(id, std::move(text));
  }
};

struct JoinTask {
  TokenizerPool& pool;
  PoolClock::time_point deadline;
  net::awaitable<Status> operator()() { co_return co_await pool.JoinUntil(deadline); }
};

StatusOr<PreparedPrompt> PrepareSync(net::io_context& io, TokenizerPool& pool,
                                     std::uint64_t id, std::string text) {
  auto future = net::co_spawn(io, PrepareTask{pool, id, std::move(text)},
                              net::use_future);
  return future.get();
}

Status JoinSync(net::io_context& io, TokenizerPool& pool,
                PoolClock::time_point deadline) {
  auto future = net::co_spawn(io, JoinTask{pool, deadline}, net::use_future);
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
  explicit PoolHarness(TokenizerPoolConfig config) : pool_(io_, std::move(config)) {
    io_thread_ = std::thread([this] { io_.run(); });
  }
  ~PoolHarness() {
    io_.stop();
    io_thread_.join();
  }

  TokenizerPool& pool() { return pool_; }
  net::io_context& io() { return io_; }
  Status StartAndWaitReady() {
    return pool_.WaitUntilReady(std::chrono::milliseconds(30000));
  }
  PoolHealth Health() const { return pool_.Health(); }

 private:
  net::io_context io_;
  TokenizerPool pool_;
  std::thread io_thread_;
};

}  // namespace

TEST(TokenizerPoolTest, DISABLED_EncodeTimingProbe) {
  auto tokenizer = inferx::Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  for (std::size_t bytes : {1u << 20, 8u << 20, 48u << 20}) {
    const auto t0 = PoolClock::now();
    const auto ids = (*tokenizer)->Encode(SlowText(bytes));
    const long ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        PoolClock::now() - t0)
                        .count();
    ASSERT_TRUE(ids.ok()) << ids.status();
    printf("[timing] %zu MiB -> %zu ids in %ld ms\n", bytes >> 20, ids->size(), ms);
    fflush(stdout);
  }
}

TEST(TokenizerPoolTest, StartupGateFailsForBadTokenizerPath) {
  TokenizerPoolConfig config = BaseConfig();
  config.tokenizer_path = "tests/testdata/tiny_tokenizer/missing.json";
  PoolHarness harness(config);
  const Status ready = harness.StartAndWaitReady();
  ASSERT_FALSE(ready.ok());
  EXPECT_EQ(ready.code(), absl::StatusCode::kFailedPrecondition);
}

TEST(TokenizerPoolTest, PrepareMatchesFacadeTokenForToken) {
  auto facade = inferx::Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(facade.ok()) << facade.status();

  PoolHarness harness(BaseConfig());
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  const std::vector<std::string> inputs = {
      "abc", "a", "abab", "hello world", "héllo 世界", "z<unk>z", std::string(4096, 'a')};
  std::uint64_t id = 1;
  for (const std::string& input : inputs) {
    const auto expected = (*facade)->Encode(input);
    ASSERT_TRUE(expected.ok()) << expected.status();
    const auto prepared = PrepareSync(harness.io(), harness.pool(), id, input);
    ASSERT_TRUE(prepared.ok()) << prepared.status();
    EXPECT_EQ(prepared->token_ids, *expected) << "input: " << input;
    EXPECT_EQ(prepared->request_id, id);
    ASSERT_NE(prepared->fingerprint, 0u);
    ++id;
  }
}

TEST(TokenizerPoolTest, OversizedPromptRejectedWithoutWorker) {
  TokenizerPoolConfig config = BaseConfig();
  config.queue_max_bytes = 1024;
  PoolHarness harness(config);
  const auto result =
      PrepareSync(harness.io(), harness.pool(), 1, SlowText(4096));
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(TokenizerPoolTest, AdmissionBoundRejectsAndThenServes) {
  auto facade = inferx::Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(facade.ok()) << facade.status();

  TokenizerPoolConfig config = BaseConfig();
  config.queue_max_requests = 1;
  config.queue_max_bytes = 64u << 20;
  PoolHarness harness(config);
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  const std::string text = SlowText(8 << 20);
  auto first = net::co_spawn(harness.io(), PrepareTask{harness.pool(), 1, text},
                             net::use_future);
  ASSERT_TRUE(WaitFor([&] { return harness.Health().workers_busy == 1; },
                      std::chrono::seconds(10)))
      << "worker never picked the job up";

  // One running job fills the request budget: the next Prepare is bounded
  // out (the HTTP layer's 503 path) without waiting on the worker.
  const auto second = PrepareSync(harness.io(), harness.pool(), 2, "ab");
  ASSERT_FALSE(second.ok());
  EXPECT_EQ(second.status().code(), absl::StatusCode::kResourceExhausted);

  const auto expected = (*facade)->Encode(text);
  ASSERT_TRUE(expected.ok()) << expected.status();
  const auto prepared = first.get();
  ASSERT_TRUE(prepared.ok()) << prepared.status();
  EXPECT_EQ(prepared->token_ids, *expected);
}

TEST(TokenizerPoolTest, CancelQueuedRequestCompletesCancelled) {
  TokenizerPoolConfig config = BaseConfig();
  config.queue_max_requests = 8;
  config.queue_max_bytes = 64u << 20;
  PoolHarness harness(config);
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  auto first = net::co_spawn(harness.io(),
                             PrepareTask{harness.pool(), 1, SlowText(8 << 20)},
                             net::use_future);
  ASSERT_TRUE(WaitFor([&] { return harness.Health().workers_busy == 1; },
                      std::chrono::seconds(10)));
  auto second = net::co_spawn(harness.io(), PrepareTask{harness.pool(), 2, "ab"},
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

TEST(TokenizerPoolTest, CancelInFlightKeepsWorkerUsable) {
  PoolHarness harness(BaseConfig());
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  auto job = net::co_spawn(harness.io(),
                           PrepareTask{harness.pool(), 7, SlowText(8 << 20)},
                           net::use_future);
  ASSERT_TRUE(WaitFor([&] { return harness.Health().workers_busy == 1; },
                      std::chrono::seconds(10)));

  harness.pool().Cancel(7);
  const auto cancelled = job.get();
  ASSERT_FALSE(cancelled.ok());
  EXPECT_EQ(cancelled.status().code(), absl::StatusCode::kCancelled);

  // The late result must be discarded and the worker must keep serving.
  const auto next = PrepareSync(harness.io(), harness.pool(), 8, "ab");
  ASSERT_TRUE(next.ok()) << next.status();
}

TEST(TokenizerPoolTest, SlowEncodeTimesOutWithDeadlineExceeded) {
  TokenizerPoolConfig config = BaseConfig();
  config.request_timeout = std::chrono::milliseconds(200);
  config.queue_max_bytes = 64u << 20;
  PoolHarness harness(config);
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  auto job = net::co_spawn(harness.io(),
                           PrepareTask{harness.pool(), 1, SlowText(8 << 20)},
                           net::use_future);
  const auto timed_out = job.get();
  ASSERT_FALSE(timed_out.ok());
  EXPECT_EQ(timed_out.status().code(), absl::StatusCode::kDeadlineExceeded);

  // The abandoned encode still finishes; accounting recovers and the pool
  // keeps serving afterwards.
  ASSERT_TRUE(WaitFor([&] { return harness.Health().queued_bytes == 0; },
                      std::chrono::seconds(10)));
  const auto next = PrepareSync(harness.io(), harness.pool(), 2, "ab");
  ASSERT_TRUE(next.ok()) << next.status();
}

TEST(TokenizerPoolTest, DrainRejectsNewWorkAndJoinsCleanly) {
  TokenizerPoolConfig config = BaseConfig();
  config.workers = 2;
  config.queue_max_bytes = 64u << 20;
  config.shutdown_grace = std::chrono::milliseconds(3000);
  PoolHarness harness(config);
  ASSERT_TRUE(harness.StartAndWaitReady().ok());

  // Slow jobs keep both workers observably busy so drain sees them admitted.
  auto first = net::co_spawn(harness.io(),
                             PrepareTask{harness.pool(), 1, SlowText(8 << 20)},
                             net::use_future);
  auto second = net::co_spawn(harness.io(),
                              PrepareTask{harness.pool(), 2, SlowText(8 << 20)},
                              net::use_future);
  ASSERT_TRUE(WaitFor([&] { return harness.Health().workers_busy == 2; },
                      std::chrono::seconds(10)));

  harness.pool().BeginDrain();
  const auto rejected = PrepareSync(harness.io(), harness.pool(), 3, "a");
  ASSERT_FALSE(rejected.ok());
  EXPECT_EQ(rejected.status().code(), absl::StatusCode::kFailedPrecondition);

  // Admitted work drains through the workers rather than being cancelled.
  const auto a = first.get();
  const auto b = second.get();
  ASSERT_TRUE(a.ok()) << a.status();
  ASSERT_TRUE(b.ok()) << b.status();

  const Status joined = JoinSync(harness.io(), harness.pool(),
                                 PoolClock::now() + std::chrono::seconds(5));
  EXPECT_TRUE(joined.ok()) << joined;
  const PoolHealth health = harness.Health();
  EXPECT_TRUE(health.draining);
  EXPECT_EQ(health.queued_requests, 0u);
}
