#pragma once

#include <folly/CancellationToken.h>
#include <folly/coro/Task.h>
#include <folly/coro/Timeout.h>
#include <folly/futures/Future.h>

#include <algorithm>
#include <chrono>
#include <utility>

#include "inferx/core/status.h"

namespace inferx::server::coroutine {

/// Awaits an operation against one absolute deadline. Folly's timeout helper
/// requests child cancellation and joins it before returning, so no detached
/// timer or operation can retain request state after this function completes.
template <typename T>
folly::coro::Task<StatusOr<T>> WithDeadline(
    folly::coro::Task<StatusOr<T>> operation,
    std::chrono::steady_clock::time_point deadline,
    folly::CancellationSource& cancellation,
    folly::Timekeeper* timekeeper = nullptr) {
  const auto now = std::chrono::steady_clock::now();
  if (deadline <= now) {
    cancellation.requestCancellation();
    co_return absl::DeadlineExceededError("request deadline exceeded");
  }
  try {
    const auto remaining = std::max(
        std::chrono::milliseconds(1),
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));
    co_return co_await folly::coro::timeout(
        folly::coro::co_withCancellation(cancellation.getToken(),
                                         std::move(operation)),
        remaining, timekeeper);
  } catch (const folly::FutureTimeout&) {
    cancellation.requestCancellation();
    co_return absl::DeadlineExceededError("request deadline exceeded");
  }
}

inline folly::coro::Task<Status> WithDeadline(
    folly::coro::Task<Status> operation,
    std::chrono::steady_clock::time_point deadline,
    folly::CancellationSource& cancellation,
    folly::Timekeeper* timekeeper = nullptr) {
  const auto now = std::chrono::steady_clock::now();
  if (deadline <= now) {
    cancellation.requestCancellation();
    co_return absl::DeadlineExceededError("request deadline exceeded");
  }
  try {
    const auto remaining = std::max(
        std::chrono::milliseconds(1),
        std::chrono::duration_cast<std::chrono::milliseconds>(deadline - now));
    co_return co_await folly::coro::timeout(
        folly::coro::co_withCancellation(cancellation.getToken(),
                                         std::move(operation)),
        remaining, timekeeper);
  } catch (const folly::FutureTimeout&) {
    cancellation.requestCancellation();
    co_return absl::DeadlineExceededError("request deadline exceeded");
  }
}

}  // namespace inferx::server::coroutine
