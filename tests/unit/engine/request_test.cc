#include "inferx/engine/request.h"

#include <gtest/gtest.h>

#include <utility>
#include <vector>

namespace inferx {
namespace {

std::vector<TokenId> Prompt(int count) {
  constexpr int kTokenIdOffset = 100;  // Arbitrary offset to avoid zero.
  std::vector<TokenId> tokens;
  tokens.reserve(count);

  for (int i = 0; i < count; ++i) {
    tokens.push_back(static_cast<TokenId>(kTokenIdOffset + i));
  }

  return tokens;
}

TEST(RequestTest, ConstructionStartsWaitingWithPromptOnly) {
  Request request(7, Prompt(3));
  EXPECT_EQ(request.GetRequestId(), 7u);
  EXPECT_EQ(request.GetStatus(), RequestStatus::kWaiting);
  EXPECT_FALSE(request.IsFinished());
  EXPECT_EQ(request.NumPromptTokens(), 3);
  EXPECT_EQ(request.NumOutputTokens(), 0);
  EXPECT_EQ(request.NumTokens(), 3);
  EXPECT_TRUE(request.Output().empty());
  EXPECT_EQ(request.NumComputedTokens(), 0);
  EXPECT_EQ(request.NumCachedTokens(), -1);  // Not scheduled yet.
  EXPECT_EQ(request.NumPreemptions(), 0);
  EXPECT_TRUE(request.CacheSalt().empty());
  EXPECT_DOUBLE_EQ(request.ArrivalTime(), 0.0);
  EXPECT_EQ(request.Prompt(), absl::MakeConstSpan(Prompt(3)));
}

TEST(RequestTest, AdmissionRequiresWaiting) {
  Request request(1, Prompt(2));
  ASSERT_TRUE(request.MarkRunning().ok());
  EXPECT_EQ(request.GetStatus(), RequestStatus::kRunning);
  // Double admission is refused.
  EXPECT_EQ(request.MarkRunning().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(RequestTest, AppendOutputRequiresRunning) {
  Request request(1, Prompt(2));
  const TokenId token = 42;
  // Waiting (never admitted) cannot accept tokens.
  EXPECT_EQ(request.AppendOutput(absl::MakeConstSpan(&token, 1)).code(),
            absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(request.MarkRunning().ok());
  ASSERT_TRUE(request.AppendOutput(absl::MakeConstSpan(&token, 1)).ok());
  EXPECT_EQ(request.NumTokens(), 3);
  EXPECT_EQ(request.NumOutputTokens(), 1);
  EXPECT_EQ(request.Output(), absl::MakeConstSpan(std::vector<TokenId>{42}));
}

TEST(RequestTest, ViewsShareTheSingleSequence) {
  Request request(1, Prompt(2));
  ASSERT_TRUE(request.MarkRunning().ok());
  ASSERT_TRUE(request.AppendOutput(absl::MakeConstSpan(std::vector<TokenId>{7, 8})).ok());
  EXPECT_EQ(request.Prompt(), absl::MakeConstSpan(Prompt(2)));
  EXPECT_EQ(request.Output(), absl::MakeConstSpan(std::vector<TokenId>{7, 8}));
  EXPECT_EQ(request.AllTokenIds().size(), 4u);
  // Prompt is the fixed prefix of the whole sequence.
  EXPECT_EQ(request.AllTokenIds().subspan(0, 2), request.Prompt());
  EXPECT_EQ(request.AllTokenIds().subspan(2), request.Output());
}

TEST(RequestTest, PreemptKeepsOutputAndResetsWatermark) {
  Request request(1, Prompt(4));
  ASSERT_TRUE(request.MarkRunning().ok());
  ASSERT_TRUE(request.AppendOutput(absl::MakeConstSpan(std::vector<TokenId>{7, 8})).ok());
  request.SetNumComputedTokens(6);

  ASSERT_TRUE(request.Preempt().ok());
  EXPECT_EQ(request.GetStatus(), RequestStatus::kPreempted);
  EXPECT_EQ(request.NumComputedTokens(), 0);  // Recompute from zero...
  EXPECT_EQ(request.NumOutputTokens(), 2);    // ...but replay keeps output (C05).
  EXPECT_EQ(request.NumPreemptions(), 1);

  // Only a running request can be preempted.
  EXPECT_EQ(request.Preempt().code(), absl::StatusCode::kFailedPrecondition);
  // Preempted requests cannot accept tokens until re-admitted.
  const TokenId token = 9;
  EXPECT_EQ(request.AppendOutput(absl::MakeConstSpan(&token, 1)).code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(RequestTest, RequeueRequiresPreempted) {
  Request request(1, Prompt(2));
  // Running -> kWaiting directly is illegal; preemption comes first.
  ASSERT_TRUE(request.MarkRunning().ok());
  EXPECT_EQ(request.MarkWaiting().code(), absl::StatusCode::kFailedPrecondition);
  ASSERT_TRUE(request.Preempt().ok());
  ASSERT_TRUE(request.MarkWaiting().ok());
  EXPECT_EQ(request.GetStatus(), RequestStatus::kWaiting);
  // The full cycle allows re-admission.
  ASSERT_TRUE(request.MarkRunning().ok());
}

TEST(RequestTest, FinishIsTerminalFromAnyNonTerminalState) {
  Request aborted(1, Prompt(2));
  ASSERT_TRUE(aborted.Finish(FinishReason::kAborted).ok());
  EXPECT_EQ(aborted.GetStatus(), RequestStatus::kFinished);
  EXPECT_TRUE(aborted.IsFinished());
  EXPECT_EQ(aborted.GetFinishReason(), FinishReason::kAborted);
  EXPECT_EQ(aborted.Finish(FinishReason::kStopped).code(),
            absl::StatusCode::kFailedPrecondition);  // Terminal, no restart.
  const TokenId token = 1;
  EXPECT_EQ(aborted.AppendOutput(absl::MakeConstSpan(&token, 1)).code(),
            absl::StatusCode::kFailedPrecondition);

  // Preempted requests can finish (aborted victims).
  Request victim(2, Prompt(2));
  ASSERT_TRUE(victim.MarkRunning().ok());
  ASSERT_TRUE(victim.Preempt().ok());
  ASSERT_TRUE(victim.Finish(FinishReason::kAborted).ok());
}

TEST(RequestTest, StopReasonRidesAlongWithFinish) {
  Request request(1, Prompt(2));
  ASSERT_TRUE(request.Finish(FinishReason::kStopped, StopReason{std::string("done")}).ok());
  const StopReason& reason = request.GetStopReason();
  ASSERT_TRUE(std::holds_alternative<std::string>(reason));
  EXPECT_EQ(std::get<std::string>(reason), "done");

  Request eos(2, Prompt(2));
  ASSERT_TRUE(eos.Finish(FinishReason::kStopped, StopReason{TokenId{99}}).ok());
  EXPECT_EQ(std::get<TokenId>(eos.GetStopReason()), 99);

  Request plain(3, Prompt(2));
  ASSERT_TRUE(plain.Finish(FinishReason::kLengthCapped).ok());
  EXPECT_TRUE(std::holds_alternative<std::monostate>(plain.GetStopReason()));
}

TEST(RequestTest, CachedTokenCountRecordsFirstWinOnly) {
  Request request(1, Prompt(8));
  request.SetNumCachedTokens(4);
  EXPECT_EQ(request.NumCachedTokens(), 4);
  // Later calls (after preemption re-adoption, say) never overwrite.
  request.SetNumCachedTokens(8);
  EXPECT_EQ(request.NumCachedTokens(), 4);
  // Negative values are ignored entirely, including as the first call.
  Request fresh(2, Prompt(8));
  fresh.SetNumCachedTokens(-1);
  EXPECT_EQ(fresh.NumCachedTokens(), -1);
  fresh.SetNumCachedTokens(4);
  EXPECT_EQ(fresh.NumCachedTokens(), 4);
}

TEST(RequestTest, MoveAssignmentTransfersState) {
  // The defaulted move assignment must exist and work: the scheduler's
  // swap-remove relies on it, and const members would have deleted it.
  Request source(1, Prompt(2));
  ASSERT_TRUE(source.MarkRunning().ok());
  ASSERT_TRUE(source.AppendOutput(absl::MakeConstSpan(std::vector<TokenId>{7})).ok());
  source.SetCacheSalt("tenant-a");
  source.SetPriority(3);

  Request target(2, Prompt(9));
  target = std::move(source);
  EXPECT_EQ(target.GetRequestId(), 1u);
  EXPECT_EQ(target.NumPromptTokens(), 2);
  EXPECT_EQ(target.NumTokens(), 3);
  EXPECT_EQ(target.GetStatus(), RequestStatus::kRunning);
  EXPECT_EQ(target.CacheSalt(), "tenant-a");
  EXPECT_EQ(target.Priority(), 3);
}

TEST(RequestTest, SamplingParamsAndBoundsComeFromParams) {
  sampling::SamplingParams params;
  params.max_tokens = 5;
  Request request(1, Prompt(2), params, 1.25);
  EXPECT_EQ(request.MaxNewTokens(), 5);
  EXPECT_EQ(request.GetSamplingParams().max_tokens, 5u);
  EXPECT_DOUBLE_EQ(request.ArrivalTime(), 1.25);
}

}  // namespace
}  // namespace inferx
