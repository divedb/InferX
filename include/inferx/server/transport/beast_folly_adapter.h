#pragma once

#include <folly/CancellationToken.h>
#include <folly/Executor.h>
#include <folly/coro/Baton.h>
#include <folly/coro/CurrentExecutor.h>
#include <folly/coro/Task.h>
#include <folly/coro/ViaIfAsync.h>

#include <atomic>
#include <boost/asio/dispatch.hpp>
#include <boost/system/error_code.hpp>
#include <exception>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

#include "inferx/core/status.h"

namespace inferx::server::transport {

namespace detail {

enum class AdapterOutcome : unsigned char { kPending, kCompleted, kCancelled };

template <typename Result>
struct AdapterState {
  std::atomic<AdapterOutcome> outcome{AdapterOutcome::kPending};
  folly::coro::Baton ready;
  std::optional<Result> result;
  boost::system::error_code error;
  std::exception_ptr exception;
};

inline Status AsioErrorStatus(const boost::system::error_code& error) {
  return absl::UnavailableError(error.message());
}

}  // namespace detail

/// Adapts one Asio-style operation to a Folly task.
///
/// `initiate` is invoked on `asio_executor` with a completion callback taking
/// `(boost::system::error_code, Result)`. `cancel` is also invoked on that
/// executor if cancellation wins. The awaiting coroutine always returns to its
/// Folly executor; it is never resumed inline on the Asio I/O thread.
///
/// Completion and cancellation race through one atomic terminal transition.
/// The shared state remains alive when a late Asio completion arrives after
/// cancellation, preventing callbacks from referencing coroutine storage.
template <typename Result, typename AsioExecutor, typename Initiate,
          typename Cancel>
folly::coro::Task<StatusOr<Result>> AwaitAsio(
    AsioExecutor asio_executor, Initiate initiate, Cancel cancel,
    folly::CancellationToken cancellation = {}) {
  static_assert(!std::is_reference_v<Result>);

  auto state = std::make_shared<detail::AdapterState<Result>>();
  auto folly_executor = co_await folly::coro::co_current_executor;

  folly::CancellationCallback cancellation_callback(
      cancellation,
      [state, asio_executor, cancel = std::move(cancel)]() mutable {
        auto expected = detail::AdapterOutcome::kPending;
        if (!state->outcome.compare_exchange_strong(
                expected, detail::AdapterOutcome::kCancelled,
                std::memory_order_acq_rel)) {
          return;
        }
        boost::asio::dispatch(asio_executor,
                              [state, cancel = std::move(cancel)]() mutable {
                                try {
                                  cancel();
                                } catch (...) {
                                  // Cancellation is best-effort. The
                                  // operation's shared state remains alive for
                                  // its eventual completion callback.
                                }
                              });
        state->ready.post();
      });

  boost::asio::dispatch(asio_executor, [state, initiate = std::move(
                                                   initiate)]() mutable {
    try {
      initiate([state](boost::system::error_code error, Result result) mutable {
        auto expected = detail::AdapterOutcome::kPending;
        if (!state->outcome.compare_exchange_strong(
                expected, detail::AdapterOutcome::kCompleted,
                std::memory_order_acq_rel)) {
          return;
        }
        state->error = error;
        state->result.emplace(std::move(result));
        state->ready.post();
      });
    } catch (...) {
      auto expected = detail::AdapterOutcome::kPending;
      if (state->outcome.compare_exchange_strong(
              expected, detail::AdapterOutcome::kCompleted,
              std::memory_order_acq_rel)) {
        state->exception = std::current_exception();
        state->ready.post();
      }
    }
  });

  co_await folly::coro::co_viaIfAsync(folly_executor, state->ready);

  if (state->outcome.load(std::memory_order_acquire) ==
      detail::AdapterOutcome::kCancelled) {
    co_return absl::CancelledError("Asio operation cancelled");
  }
  if (state->exception) {
    try {
      std::rethrow_exception(state->exception);
    } catch (const std::exception& error) {
      co_return absl::InternalError(error.what());
    } catch (...) {
      co_return absl::InternalError("Asio operation initiation failed");
    }
  }
  if (state->error) {
    co_return detail::AsioErrorStatus(state->error);
  }
  co_return std::move(*state->result);
}

}  // namespace inferx::server::transport
