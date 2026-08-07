#include "inferx/server/transport/beast_session.h"

#include <boost/asio/buffer.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/write.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <folly/CancellationToken.h>
#include <folly/Try.h>
#include <folly/coro/Task.h>

#include <atomic>
#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include "inferx/server/transport/beast_folly_adapter.h"

namespace inferx::server::transport {
namespace {

namespace asio = boost::asio;
namespace beast = boost::beast;
namespace http = beast::http;

class BeastResponseWriter final : public ResponseWriter {
 public:
  BeastResponseWriter(beast::tcp_stream* stream, int write_timeout_seconds)
      : stream_(stream), write_timeout_seconds_(write_timeout_seconds) {}

  folly::coro::Task<Status> WriteResponse(
      HttpResponse response, folly::CancellationToken cancellation) override {
    if (!Begin(State::kResponse)) {
      co_return FailedPreconditionError("response has already started");
    }
    auto message = std::make_shared<HttpResponse>(std::move(response));
    stream_->expires_after(std::chrono::seconds(write_timeout_seconds_));
    auto result = co_await AwaitAsio<size_t>(
        stream_->get_executor(),
        [stream = stream_, message](auto completion) mutable {
          http::async_write(
              *stream, *message,
              [message, completion = std::move(completion)](
                  beast::error_code error, size_t bytes) mutable {
                completion(error, bytes);
              });
        },
        [stream = stream_] { CancelSocket(*stream); }, cancellation);
    state_ = result.ok() ? State::kFinished : State::kFailed;
    co_return result.ok() ? OkStatus() : result.status();
  }

  folly::coro::Task<Status> StartStream(
      HttpStreamHead head, folly::CancellationToken cancellation) override {
    if (!Begin(State::kStream)) {
      co_return FailedPreconditionError("response has already started");
    }
    auto message = std::make_shared<HttpStreamHead>(std::move(head));
    auto serializer =
        std::make_shared<http::response_serializer<http::empty_body>>(*message);
    stream_->expires_after(std::chrono::seconds(write_timeout_seconds_));
    auto result = co_await AwaitAsio<size_t>(
        stream_->get_executor(),
        [stream = stream_, message, serializer](auto completion) mutable {
          http::async_write_header(
              *stream, *serializer,
              [message, serializer, completion = std::move(completion)](
                  beast::error_code error, size_t bytes) mutable {
                completion(error, bytes);
              });
        },
        [stream = stream_] { CancelSocket(*stream); }, cancellation);
    if (!result.ok()) state_ = State::kFailed;
    co_return result.ok() ? OkStatus() : result.status();
  }

  folly::coro::Task<Status> Write(
      std::string data, folly::CancellationToken cancellation) override {
    if (state_ != State::kStream) {
      co_return FailedPreconditionError("stream has not started");
    }
    auto payload = std::make_shared<std::string>(std::move(data));
    stream_->expires_after(std::chrono::seconds(write_timeout_seconds_));
    auto result = co_await AwaitAsio<size_t>(
        stream_->get_executor(),
        [stream = stream_, payload](auto completion) mutable {
          asio::async_write(
              *stream, http::make_chunk(asio::buffer(*payload)),
              [payload, completion = std::move(completion)](
                  beast::error_code error, size_t bytes) mutable {
                completion(error, bytes);
              });
        },
        [stream = stream_] { CancelSocket(*stream); }, cancellation);
    if (!result.ok()) state_ = State::kFailed;
    co_return result.ok() ? OkStatus() : result.status();
  }

  folly::coro::Task<Status> Finish(
      folly::CancellationToken cancellation) override {
    if (state_ != State::kStream) {
      co_return FailedPreconditionError("stream has not started");
    }
    stream_->expires_after(std::chrono::seconds(write_timeout_seconds_));
    auto result = co_await AwaitAsio<size_t>(
        stream_->get_executor(),
        [stream = stream_](auto completion) mutable {
          asio::async_write(
              *stream, http::make_chunk_last(),
              [completion = std::move(completion)](
                  beast::error_code error, size_t bytes) mutable {
                completion(error, bytes);
              });
        },
        [stream = stream_] { CancelSocket(*stream); }, cancellation);
    state_ = result.ok() ? State::kFinished : State::kFailed;
    co_return result.ok() ? OkStatus() : result.status();
  }

  bool finished() const { return state_ == State::kFinished; }

 private:
  enum class State { kIdle, kResponse, kStream, kFinished, kFailed };

  static void CancelSocket(beast::tcp_stream& stream) {
    beast::error_code ignored;
    stream.socket().cancel(ignored);
  }

  bool Begin(State state) {
    if (state_ != State::kIdle) return false;
    state_ = state;
    return true;
  }

  beast::tcp_stream* stream_;
  int write_timeout_seconds_;
  State state_ = State::kIdle;
};

}  // namespace

class BeastSession : public std::enable_shared_from_this<BeastSession> {
 public:
  BeastSession(asio::ip::tcp::socket socket, BeastListenerConfig config,
               std::shared_ptr<RequestHandler> handler,
               folly::Executor::KeepAlive<> coroutine_executor)
      : stream_(std::move(socket)),
        config_(std::move(config)),
        handler_(std::move(handler)),
        coroutine_executor_(std::move(coroutine_executor)) {}

  folly::coro::Task<void> Run() {
    beast::error_code option_error;
    stream_.socket().set_option(asio::ip::tcp::no_delay(true), option_error);
    bool first_request = true;
    while (!cancel_.isCancellationRequested()) {
      auto parser =
          std::make_shared<http::request_parser<http::string_body>>();
      parser->body_limit(config_.max_request_bytes);
      stream_.expires_after(
          std::chrono::seconds(first_request
                                   ? config_.header_read_timeout_seconds
                                   : config_.keep_alive_timeout_seconds));
      auto header = co_await AwaitAsio<size_t>(
          stream_.get_executor(),
          [this, parser](auto completion) mutable {
            http::async_read_header(
                stream_, buffer_, *parser,
                [parser, completion = std::move(completion)](
                    beast::error_code error, size_t bytes) mutable {
                  completion(error, bytes);
                });
          },
          [this] { CancelSocket(); }, cancel_.getToken());
      if (!header.ok()) {
        const std::string message(header.status().message());
        const bool body_limit = message.find("body limit") != std::string::npos;
        const bool malformed = message.find("bad") != std::string::npos;
        if (body_limit || malformed) {
          BeastResponseWriter writer(&stream_, config_.write_timeout_seconds);
          HttpResponse response{body_limit ? http::status::payload_too_large
                                           : http::status::bad_request,
                                11};
          response.set(http::field::content_type, "application/json");
          response.keep_alive(false);
          response.body() = body_limit
                                ? "{\"error\":{\"message\":\"request body "
                                  "too large\",\"type\":\"invalid_request_error\"}}"
                                : "{\"error\":{\"message\":\"malformed HTTP "
                                  "request\",\"type\":\"invalid_request_error\"}}";
          response.prepare_payload();
          (void)co_await writer.WriteResponse(std::move(response),
                                               cancel_.getToken());
        }
        break;
      }
      first_request = false;

      if (!parser->is_done()) {
        stream_.expires_after(
            std::chrono::seconds(config_.body_read_timeout_seconds));
        auto body = co_await AwaitAsio<size_t>(
            stream_.get_executor(),
            [this, parser](auto completion) mutable {
              http::async_read(
                  stream_, buffer_, *parser,
                  [parser, completion = std::move(completion)](
                      beast::error_code error, size_t bytes) mutable {
                    completion(error, bytes);
                  });
            },
            [this] { CancelSocket(); }, cancel_.getToken());
        if (!body.ok()) {
          const bool body_limit =
              std::string(body.status().message()).find("body limit") !=
              std::string::npos;
          if (body_limit) {
            BeastResponseWriter writer(&stream_, config_.write_timeout_seconds);
            HttpResponse response{http::status::payload_too_large, 11};
            response.set(http::field::content_type, "application/json");
            response.keep_alive(false);
            response.body() =
                "{\"error\":{\"message\":\"request body too large\","
                "\"type\":\"invalid_request_error\"}}";
            response.prepare_payload();
            (void)co_await writer.WriteResponse(std::move(response),
                                                 cancel_.getToken());
          }
          break;
        }
      }

      HttpRequest request = parser->release();
      const bool keep_alive = request.keep_alive();
      BeastResponseWriter writer(&stream_, config_.write_timeout_seconds);
      bool handler_failed = false;
      try {
        co_await handler_->Handle(std::move(request), RequestContext{}, writer,
                                  cancel_.getToken());
      } catch (...) {
        handler_failed = true;
      }
      if (handler_failed) break;
      if (!writer.finished() || !keep_alive) break;
    }
    Close();
  }

  void Stop() {
    cancel_.requestCancellation();
    asio::dispatch(stream_.get_executor(), [self = shared_from_this()] {
      self->Close();
    });
  }

  folly::Executor::KeepAlive<> coroutine_executor() const {
    return coroutine_executor_;
  }

 private:
  void CancelSocket() {
    beast::error_code ignored;
    stream_.socket().cancel(ignored);
  }

  void Close() {
    if (closed_.exchange(true, std::memory_order_acq_rel)) return;
    beast::error_code ignored;
    stream_.socket().shutdown(asio::ip::tcp::socket::shutdown_both, ignored);
    stream_.socket().close(ignored);
  }

  beast::tcp_stream stream_;
  beast::flat_buffer buffer_;
  BeastListenerConfig config_;
  std::shared_ptr<RequestHandler> handler_;
  folly::Executor::KeepAlive<> coroutine_executor_;
  folly::CancellationSource cancel_;
  std::atomic<bool> closed_{false};
};

namespace {

folly::coro::Task<void> RunOwnedSession(
    std::shared_ptr<BeastSession> session) {
  co_await session->Run();
}

}  // namespace

std::shared_ptr<BeastSession> MakeBeastSession(
    asio::ip::tcp::socket socket, BeastListenerConfig config,
    std::shared_ptr<RequestHandler> handler,
    folly::Executor::KeepAlive<> coroutine_executor) {
  return std::make_shared<BeastSession>(
      std::move(socket), std::move(config), std::move(handler),
      std::move(coroutine_executor));
}

void StartBeastSession(const std::shared_ptr<BeastSession>& session) {
  folly::coro::co_withExecutor(session->coroutine_executor(),
                               RunOwnedSession(session))
      .start([](folly::Try<void>&&) {});
}

void StopBeastSession(const std::shared_ptr<BeastSession>& session) {
  session->Stop();
}

}  // namespace inferx::server::transport
