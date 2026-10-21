/// \brief Single-slot awaitable mailbox for worker-to-coroutine results.
///
/// The delivery pattern of the prompt pool and detokenizer (the same one
/// EngineQueue uses): a worker thread completes the slot; the awaiting
/// coroutine on an Asio executor takes it. The buffered-token
/// concurrent_channel closes the check-then-wait race — a try_send with no
/// parked receiver is buffered, so delivery is independent of the
/// arm-vs-complete interleaving. Complete is exactly-once; late deliveries
/// after cancel/timeout are no-ops whose results are simply never taken.
#ifndef INFERX_SERVER_RESULT_MAILBOX_H_
#define INFERX_SERVER_RESULT_MAILBOX_H_

#include <mutex>
#include <optional>
#include <utility>

#include <boost/asio.hpp>

#include "inferx/core/status.h"

namespace inferx {

template <typename T>
class ResultMailbox {
 public:
  explicit ResultMailbox(boost::asio::any_io_executor executor)
      : wake_(executor, 1) {}

  /// Completes the slot; any thread. Exactly-once: later calls are no-ops.
  void Complete(StatusOr<T> result) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (delivered_) return;
      delivered_ = true;
      result_ = std::move(result);
    }
    wake_.try_send(std::exception_ptr{});
  }

  /// Returns the completed result; runs on the awaiting coroutine's
  /// executor. Safe to destroy while parked (the channel absorbs the token).
  boost::asio::awaitable<StatusOr<T>> Take() {
    for (;;) {
      {
        std::lock_guard<std::mutex> lock(mu_);
        if (delivered_) co_return std::move(*result_);
      }
      auto [eptr] = co_await wake_.async_receive(
          boost::asio::as_tuple(boost::asio::use_awaitable));
      if (eptr) co_return InternalError("result mailbox closed");
    }
  }

 private:
  std::mutex mu_;
  bool delivered_ = false;
  std::optional<StatusOr<T>> result_;
  boost::asio::experimental::concurrent_channel<void(std::exception_ptr)> wake_;
};

}  // namespace inferx

#endif  // INFERX_SERVER_RESULT_MAILBOX_H_
