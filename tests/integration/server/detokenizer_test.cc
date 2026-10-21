// Incremental detokenizer tests (docs/tokenizer_design.md §8.2, §12.2):
// concatenated deltas must equal the full decode (no duplicated prefix),
// no delta may end inside a UTF-8 character, the surrounding window must
// slide over long generations without corrupting output, and Finish/Cancel
// release state exactly once.
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <gtest/gtest.h>

#include "inferx/server/detokenizer.h"
#include "inferx/server/utf8.h"
#include "inferx/tokenizer/tokenizer.h"

namespace net = boost::asio;
using inferx::Tokenizer;
using inferx::DecodeDelta;
using inferx::Detokenizer;

namespace {

constexpr const char* kTokenizerPath =
    "tests/testdata/tiny_tokenizer/tokenizer.json";

/// Coroutine wrappers as functors: lambdas cannot be coroutines portably.
struct AppendTask {
  Detokenizer& detokenizer;
  std::uint64_t id;
  std::vector<inferx::TokenId> tokens;
  net::awaitable<inferx::StatusOr<DecodeDelta>> operator()() {
    co_return co_await detokenizer.Append(id, std::move(tokens), {});
  }
};

struct FinishTask {
  Detokenizer& detokenizer;
  std::uint64_t id;
  net::awaitable<inferx::StatusOr<DecodeDelta>> operator()() {
    co_return co_await detokenizer.Finish(id);
  }
};

class DetokenizerHarness {
 public:
  DetokenizerHarness()
      : work_(net::make_work_guard(io_.get_executor())),
        detokenizer_(inferx::CreateDetokenizer(kTokenizerPath)) {
    // Keep the io_context alive between jobs: unlike the prompt pool, the
    // detokenizer posts nothing at construction, so run() would otherwise
    // return before the first Append.
    io_thread_ = std::thread([this] { io_.run(); });
  }
  ~DetokenizerHarness() {
    io_.stop();
    io_thread_.join();
  }

  Detokenizer& detokenizer() { return *detokenizer_; }
  net::io_context& io() { return io_; }

  inferx::StatusOr<DecodeDelta> Append(std::uint64_t id, std::vector<inferx::TokenId> tokens) {
    auto future = net::co_spawn(io_, AppendTask{detokenizer(), id, std::move(tokens)},
                                net::use_future);
    return future.get();
  }

  inferx::StatusOr<DecodeDelta> Finish(std::uint64_t id) {
    auto future = net::co_spawn(io_, FinishTask{detokenizer(), id}, net::use_future);
    return future.get();
  }

 private:
  net::io_context io_;
  std::optional<boost::asio::executor_work_guard<net::io_context::executor_type>> work_;
  std::shared_ptr<Detokenizer> detokenizer_;
  std::thread io_thread_;
};

/// Feeds `ids` to Append one token at a time, returns the concatenated
/// deltas including the Finish flush.
std::string StreamOneAtATime(DetokenizerHarness& harness, std::uint64_t id,
                             const std::vector<inferx::TokenId>& ids) {
  std::string streamed;
  for (const inferx::TokenId token : ids) {
    auto delta = harness.Append(id, {token});
    if (!delta.ok()) return streamed;
    streamed += delta->text;
  }
  auto tail = harness.Finish(id);
  EXPECT_TRUE(tail.ok()) << tail.status();
  streamed += tail->text;
  return streamed;
}

}  // namespace

TEST(DetokenizerTest, DeltasConcatenateToFullDecode) {
  auto tokenizer = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  DetokenizerHarness harness;

  const std::vector<std::string> texts = {
      "hello world", "héllo 世界", "a", "aaaa bbbb cccc dddd",
      "multi byte: 日本語テキスト、絵文字 😀 fin"};
  for (const std::string& text : texts) {
    const auto ids = (*tokenizer)->Encode(text);
    ASSERT_TRUE(ids.ok()) << ids.status();
    const auto full = (*tokenizer)->Decode(*ids);
    ASSERT_TRUE(full.ok()) << full.status();
    EXPECT_EQ(StreamOneAtATime(harness, 101, *ids), *full) << "text: " << text;
  }
}

TEST(DetokenizerTest, NoDeltaEndsInsideAUtf8Character) {
  auto tokenizer = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  DetokenizerHarness harness;

  const auto ids = (*tokenizer)->Encode("héllo 世界 😀 😀 😀");
  ASSERT_TRUE(ids.ok()) << ids.status();
  for (const inferx::TokenId token : *ids) {
    auto delta = harness.Append(202, {token});
    ASSERT_TRUE(delta.ok()) << delta.status();
    EXPECT_EQ(inferx::detail::Utf8Boundary(delta->text), delta->text.size())
        << "delta splits a UTF-8 character: " << delta->text;
  }
  EXPECT_TRUE(harness.Finish(202).ok());
}

TEST(DetokenizerTest, LongStreamsMatchFullDecode) {
  auto tokenizer = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  DetokenizerHarness harness;

  // Long past any plausible surrounding-window size: whatever decode
  // anchoring the implementation uses, suffixes must keep concatenating to
  // the full decode.
  std::string text;
  for (int i = 0; i < 120; ++i) text += "the quick brown fox jumps ";
  text += "héllo 世界";
  const auto ids = (*tokenizer)->Encode(text);
  ASSERT_TRUE(ids.ok()) << ids.status();
  ASSERT_GT(ids->size(), 128u) << "fixture tokenizer split too coarsely for this test";

  const auto full = (*tokenizer)->Decode(*ids);
  ASSERT_TRUE(full.ok()) << full.status();
  EXPECT_EQ(StreamOneAtATime(harness, 303, *ids), *full);
}

TEST(DetokenizerTest, MultiTokenAppendBatchesMatchTokenAtATime) {
  auto tokenizer = Tokenizer::FromFile(kTokenizerPath);
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  DetokenizerHarness harness;

  const auto ids = (*tokenizer)->Encode("streaming in batches: héllo 世界, more text");
  ASSERT_TRUE(ids.ok()) << ids.status();
  const auto full = (*tokenizer)->Decode(*ids);
  ASSERT_TRUE(full.ok()) << full.status();

  std::string streamed;
  for (std::size_t i = 0; i < ids->size(); i += 3) {
    std::vector<inferx::TokenId> batch(ids->begin() + i,
                                       ids->begin() + std::min(i + 3, ids->size()));
    auto delta = harness.Append(404, std::move(batch));
    ASSERT_TRUE(delta.ok()) << delta.status();
    streamed += delta->text;
  }
  streamed += harness.Finish(404)->text;
  EXPECT_EQ(streamed, *full);
}

TEST(DetokenizerTest, CancelAndFinishReleaseState) {
  DetokenizerHarness harness;

  auto first = harness.Append(505, {1, 2, 3});
  ASSERT_TRUE(first.ok()) << first.status();

  harness.detokenizer().Cancel(505);
  // Finish after cancel reports an empty tail, not an error.
  const auto tail = harness.Finish(505);
  ASSERT_TRUE(tail.ok()) << tail.status();
  EXPECT_TRUE(tail->text.empty());

  // Finish of an unknown id is likewise an empty delta.
  const auto unknown = harness.Finish(999);
  ASSERT_TRUE(unknown.ok()) << unknown.status();
  EXPECT_TRUE(unknown->text.empty());
}

TEST(DetokenizerTest, UnreadyDetokenizerReportsUnavailable) {
  auto detokenizer = inferx::CreateDetokenizer(
      "tests/testdata/tiny_tokenizer/missing.json");
  net::io_context io;
  auto future = net::co_spawn(
      io, AppendTask{*detokenizer, 1, {1, 2}}, net::use_future);
  io.run();
  const auto result = future.get();
  ASSERT_FALSE(result.ok());
  EXPECT_EQ(result.status().code(), absl::StatusCode::kUnavailable);
}
