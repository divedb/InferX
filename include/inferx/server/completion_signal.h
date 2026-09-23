// One-shot, thread-safe completion signal with broadcast for any number of
// coroutine waiters. Each waiter registers its own one-slot channel; Fire
// delivers a token to every registered channel. A try_send into a channel
// with no parked receiver is buffered, so delivery is independent of
// arm-vs-fire interleaving — the check-then-wait race is closed.
#ifndef INFERX_SERVER_COMPLETION_SIGNAL_H_
#define INFERX_SERVER_COMPLETION_SIGNAL_H_

#include <memory>
#include <mutex>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <boost/asio/experimental/concurrent_channel.hpp>

namespace inferx::server {

class CompletionSignal {
 public:
  /// \brief Fires once; later calls are no-ops. Any thread.
  void Fire() {
    std::vector<std::shared_ptr<Waiter>> waiters;
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (fired_) return;
      fired_ = true;
      waiters = std::move(waiters_);
      waiters_.clear();
    }
    for (const auto& waiter : waiters) {
      waiter->wake.try_send(std::exception_ptr{});
    }
  }

  bool Fired() const {
    std::lock_guard<std::mutex> lock(mu_);
    return fired_;
  }

  /// \brief Returns when fired. Safe for many concurrent waiters and for
  /// destroying an awaiting coroutine (its channel may simply absorb the
  /// token unread).
  boost::asio::awaitable<void> Wait() {
    if (Fired()) co_return;
    auto waiter = std::make_shared<Waiter>(
        co_await boost::asio::this_coro::executor);
    {
      std::lock_guard<std::mutex> lock(mu_);
      if (fired_) co_return;
      waiters_.push_back(waiter);
    }
    co_await waiter->wake.async_receive(
        boost::asio::as_tuple(boost::asio::use_awaitable));
    co_return;
  }

 private:
  struct Waiter {
    explicit Waiter(boost::asio::any_io_executor executor) : wake(executor, 1) {}
    boost::asio::experimental::concurrent_channel<void(std::exception_ptr)> wake;
  };

  mutable std::mutex mu_;
  bool fired_ = false;
  std::vector<std::shared_ptr<Waiter>> waiters_;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_COMPLETION_SIGNAL_H_
