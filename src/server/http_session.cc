#include "inferx/server/connection.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <utility>

#include <boost/asio/bind_executor.hpp>
#include <boost/asio/buffer.hpp>
#include <boost/asio/co_spawn.hpp>
#include <boost/asio/dispatch.hpp>
#include <boost/asio/post.hpp>
#include <boost/asio/redirect_error.hpp>
#include <boost/asio/use_awaitable.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/core/flat_buffer.hpp>
#include <boost/beast/core/tcp_stream.hpp>
#include <boost/beast/http/error.hpp>
#include <boost/beast/http/message.hpp>
#include <boost/beast/http/parser.hpp>
#include <boost/beast/http/read.hpp>
#include <boost/beast/http/vector_body.hpp>
#include <boost/beast/http/write.hpp>

#include "inferx/server/api_error_mapper.h"
#include "inferx/server/http_types.h"
#include "inferx/server/request_context.h"
#include "inferx/server/sse_writer.h"
#include "inferx/server/status.h"

namespace inferx::server {

namespace beast = boost::beast;

namespace {

namespace http = boost::beast::http;
namespace net = boost::asio;

using RequestParser = http::request_parser<http::vector_body<std::byte>>;
using Response = http::response<http::vector_body<std::byte>>;

/// \brief Percent-decodes a path; returns false on malformed input or
///        embedded control characters.
bool DecodeUrlPath(const std::string& raw, std::string& out) {
  out.clear();
  out.reserve(raw.size());
  for (std::size_t i = 0; i < raw.size(); ++i) {
    if (raw[i] != '%') {
      const auto byte = static_cast<unsigned char>(raw[i]);
      if (byte < 0x20 || byte == 0x7F) return false;
      out.push_back(raw[i]);
      continue;
    }
    if (i + 2 >= raw.size()) return false;
    const auto hex = [](char c) -> int {
      if (c >= '0' && c <= '9') return c - '0';
      if (c >= 'a' && c <= 'f') return c - 'a' + 10;
      if (c >= 'A' && c <= 'F') return c - 'A' + 10;
      return -1;
    };
    const int hi = hex(raw[i + 1]);
    const int lo = hex(raw[i + 2]);
    if (hi < 0 || lo < 0) return false;
    i += 2;
    out.push_back(static_cast<char>((hi << 4) | lo));
  }
  return true;
}

Response ToBeastResponse(HttpResponse&& in, bool keep_alive,
                         bool head_only) {
  Response response{static_cast<http::status>(in.status), 11};
  // header<false, Fields> derives from Fields; assign through the base.
  static_cast<http::fields&>(response.base()) = std::move(in.headers);
  if (head_only) {
    // Preserve the declared length; the serializer omits the body for HEAD.
    response.set(http::field::content_length, std::to_string(in.body.size()));
  } else {
    response.body() = std::move(in.body);
  }
  response.keep_alive(keep_alive);
  if (!head_only) response.prepare_payload();
  return response;
}

/// \brief Converts a parsed Beast message into the normalized HttpRequest.
Result<HttpRequest> BuildRequest(
    http::request<http::vector_body<std::byte>>&& message,
    const ConnectionContext& context) {
  HttpRequest request;
  if (!IsSupportedMethod(message.method())) {
    return ApiErrorMapper::MalformedRequest("unsupported HTTP method");
  }
  request.method = message.method();
  request.target = std::string(message.target());
  const std::size_t query_pos = request.target.find('?');
  const std::string raw_path =
      query_pos == std::string::npos
          ? request.target
          : request.target.substr(0, query_pos);
  request.query = query_pos == std::string::npos
                      ? std::string()
                      : request.target.substr(query_pos + 1);
  if (!DecodeUrlPath(raw_path, request.path)) {
    return ApiErrorMapper::MalformedRequest("malformed request target");
  }
  // keep_alive is derived from the Connection field, so read it before the
  // fields are moved out of the message.
  const bool keep_alive = message.keep_alive();
  request.headers = std::move(static_cast<http::fields&>(message.base()));
  request.body = std::move(message.body());
  request.keep_alive = keep_alive;
  request.peer_ip = context.peer_ip;
  request.peer_port = context.peer_port;
  return request;
}

}  // namespace

struct HttpSession::Impl {
  Impl(beast::tcp_stream&& stream_in, ConnectionContext context_in,
       ServerConfig config_in, SessionServices services_in,
       ConnectionRegistry& registry_in)
      : stream(std::move(stream_in)),
        context(std::move(context_in)),
        config(std::move(config_in)),
        services(std::move(services_in)),
        registry(registry_in),
        writer(stream.get_executor(), stream, config.timeouts.stream_write) {
    stream.expires_never();
  }

  beast::tcp_stream stream;
  ConnectionContext context;
  ServerConfig config;
  SessionServices services;
  ConnectionRegistry& registry;
  SseWriter writer;
  beast::flat_buffer buffer;
  std::atomic<bool> draining{false};
  std::atomic<bool> closed{false};
  bool processing = false;  // strand-confined: request in flight
  std::uint64_t requests_served = 0;
};

HttpSession::HttpSession(beast::tcp_stream&& stream, ConnectionContext context,
                         ServerConfig config, SessionServices services,
                         ConnectionRegistry& registry)
    : impl_(std::make_unique<Impl>(std::move(stream), std::move(context),
                                   std::move(config), std::move(services),
                                   registry)) {}

HttpSession::~HttpSession() {
  boost::system::error_code ec;
  impl_->stream.socket().close(ec);
}

ConnectionId HttpSession::id() const noexcept { return impl_->context.id; }

void HttpSession::Start() {
  auto self = shared_from_this();
  net::co_spawn(
      impl_->stream.get_executor(),
      [self]() -> net::awaitable<void> { co_await self->Run(); },
      [self](std::exception_ptr error) {
        if (error) {
          // Mapped and dropped: exceptions never reach io_context::run().
          (void)ApiErrorMapper::FromException(error, ErrorOrigin::kInternal);
        }
        self->impl_->registry.Remove(self->id());
      });
}

void HttpSession::PostStrand(std::function<void()> action) noexcept {
  auto self = shared_from_this();
  net::post(impl_->stream.get_executor(),
            [self, action = std::move(action)]() { action(); });
}

void HttpSession::BeginDrain() noexcept {
  impl_->draining.store(true, std::memory_order_release);
  PostStrand([this]() {
    // Active requests finish and then the loop exits; an idle session is
    // cancelled out of its keep-alive read immediately.
    if (!impl_->processing) {
      boost::system::error_code ec;
      impl_->stream.socket().cancel(ec);
    }
  });
}

void HttpSession::Close() noexcept {
  if (impl_->closed.exchange(true, std::memory_order_acq_rel)) return;
  PostStrand([this]() {
    boost::system::error_code ec;
    impl_->stream.socket().cancel(ec);
    impl_->stream.socket().shutdown(net::socket_base::shutdown_both, ec);
    impl_->stream.socket().close(ec);
  });
}

net::awaitable<void> HttpSession::Run() {
  struct ProcessingGuard {
    bool& flag;
    ~ProcessingGuard() { flag = false; }
  };

  for (;;) {
    if (impl_->draining.load(std::memory_order_acquire) ||
        impl_->closed.load(std::memory_order_acquire)) {
      break;
    }
    if (impl_->requests_served >=
        impl_->config.http.max_requests_per_connection) {
      break;
    }

    // ---- Read request head ----
    RequestParser parser;
    parser.header_limit(static_cast<std::uint32_t>(
        impl_->config.http.max_request_line_bytes +
        impl_->config.http.max_header_bytes));
    parser.body_limit(impl_->config.http.max_body_bytes);

    const bool first_request = impl_->requests_served == 0;
    impl_->stream.expires_after(
        first_request ? impl_->config.timeouts.header_read
                      : impl_->config.timeouts.keep_alive);
    boost::system::error_code ec;
    co_await http::async_read_header(
        impl_->stream, impl_->buffer, parser,
        net::redirect_error(net::use_awaitable, ec));
    if (ec == http::error::end_of_stream) {
      break;  // clean keep-alive close
    }
    if (ec == beast::error::timeout) {
      if (!first_request) break;  // idle keep-alive expiry: silent close
      co_await WriteEarlyError(ApiErrorMapper::RequestTimeout());
      break;
    }
    if (ec == http::error::header_limit) {
      co_await WriteEarlyError(ApiErrorMapper::HeadersTooLarge());
      break;
    }
    if (ec == http::error::body_limit) {
      // Declared Content-Length already exceeds the cap: reject without
      // reading the body.
      co_await WriteEarlyError(ApiErrorMapper::PayloadTooLarge());
      break;
    }
    if (ec) {
      co_await WriteEarlyError(
          ApiErrorMapper::MalformedRequest("malformed HTTP request head"));
      break;
    }

    const auto& message = parser.get();
    if (message.target().size() > impl_->config.http.max_url_bytes) {
      co_await WriteEarlyError(ApiErrorMapper::MalformedRequest(
          "request target exceeds the configured limit"));
      break;
    }
    std::size_t header_count = 0;
    for (const auto& field : message) {
      (void)field;
      ++header_count;
    }
    if (header_count > impl_->config.http.max_header_count) {
      co_await WriteEarlyError(ApiErrorMapper::HeadersTooLarge());
      break;
    }

    // ---- Read body (bounded) ----
    impl_->stream.expires_after(impl_->config.timeouts.body_read);
    ec = {};
    co_await http::async_read(impl_->stream, impl_->buffer, parser,
                              net::redirect_error(net::use_awaitable, ec));
    if (ec == http::error::body_limit) {
      co_await WriteEarlyError(ApiErrorMapper::PayloadTooLarge());
      break;
    }
    if (ec == beast::error::timeout) {
      co_await WriteEarlyError(ApiErrorMapper::RequestTimeout());
      break;
    }
    if (ec) {
      co_await WriteEarlyError(
          ApiErrorMapper::MalformedRequest("truncated request body"));
      break;
    }

    impl_->processing = true;
    ProcessingGuard processing_guard{impl_->processing};

    // ---- Build the normalized request ----
    auto parsed_request =
        BuildRequest(parser.release(), impl_->context);
    if (!parsed_request) {
      co_await WriteEarlyError(parsed_request.error());
      break;
    }
    HttpRequest request = std::move(parsed_request.value());

    if (impl_->draining.load(std::memory_order_acquire) ||
        impl_->closed.load(std::memory_order_acquire)) {
      // A request read after drain began is rejected, not served.
      co_await WriteEarlyError(ApiErrorMapper::ShuttingDown());
      break;
    }

    // ---- Dispatch through middleware + router on a request context ----
    ++impl_->requests_served;
    const auto received_at = Clock::now();
    // Resolve the ID before moving the request: argument evaluation order
    // is unspecified and GCC initializes parameters right-to-left.
    const ClientRequestId client_id = ResolveClientRequestId(request.headers);
    RequestContext ctx(client_id, impl_->context.id, TenantId{},
                       std::move(request), impl_->stream.get_executor(),
                       received_at,
                       received_at + impl_->config.timeouts.request_total,
                       &impl_->writer);
    const bool head_only = ctx.request().method == HttpMethod::head;

    HttpResponse response;
    bool stream_completed = false;
    try {
      // Admission (local limits + drain gate) runs before middleware so
      // rejected requests never reach handlers (design §4.1 ordering).
      // Health probes are exempt: /health, /livez, and /readyz must stay
      // answerable (with their own semantics) while draining.
      const std::string& request_path = ctx.request().path;
      const bool is_health_probe = request_path == "/health" ||
                                   request_path == "/livez" ||
                                   request_path == "/readyz";
      if (impl_->services.admission && !is_health_probe) {
        AdmissionRequest admission_request;
        admission_request.tenant = TenantId{};
        admission_request.body_bytes = ctx.request().body.size();
        admission_request.route_operation_id = ctx.request().path;
        auto lease = co_await impl_->services.admission->Acquire(
            admission_request, ctx);
        if (!lease) {
          if (impl_->services.metrics) {
            impl_->services.metrics->ObserveAdmissionRejected(
                lease.error().code);
          }
          ctx.finalize(RequestPhase::kCancelled);
          co_await WriteResponse(
              ApiErrorMapper::ToHttpResponse(lease.error()), false, head_only);
          break;
        }
      }
      const std::shared_ptr<Router> router = impl_->services.router;
      auto terminal = [&ctx, router]() -> net::awaitable<MiddlewareOutcome> {
        RouteResult result = co_await router->Dispatch(ctx);
        MiddlewareOutcome outcome;
        if (std::holds_alternative<HttpResponse>(result)) {
          outcome = std::move(std::get<HttpResponse>(result));
        } else if (std::holds_alternative<StreamCompleted>(result)) {
          outcome = MiddlewareContinue{};  // response already streamed
        } else {
          outcome = std::move(std::get<ApiError>(result));
        }
        co_return outcome;
      };
      MiddlewareOutcome outcome =
          co_await impl_->services.middleware->Run(ctx, std::move(terminal));
      if (std::holds_alternative<HttpResponse>(outcome)) {
        response = std::move(std::get<HttpResponse>(outcome));
      } else if (std::holds_alternative<ApiError>(outcome)) {
        response = ApiErrorMapper::ToHttpResponse(
            std::get<ApiError>(std::move(outcome)));
      } else {
        // Streaming already wrote and finalized the response.
        stream_completed = true;
      }
    } catch (...) {
      response = ApiErrorMapper::ToHttpResponse(ApiErrorMapper::FromException(
          std::current_exception(), ErrorOrigin::kInternal));
    }
    if (ctx.phase() != RequestPhase::kCompleted &&
        ctx.phase() != RequestPhase::kCancelled &&
        ctx.phase() != RequestPhase::kFailed) {
      ctx.finalize(response.status < 400 ? RequestPhase::kCompleted
                                         : RequestPhase::kFailed);
    }

    if (stream_completed) {
      // The stream consumed the connection; it cannot return to keep-alive.
      break;
    }

    if (impl_->services.metrics) {
      HttpRequestMetrics metrics;
      metrics.route = ctx.request().path;
      metrics.method = std::string(http::to_string(ctx.request().method));
      metrics.status = response.status;
      metrics.request_bytes = ctx.request().body.size();
      metrics.response_bytes = response.body.size();
      impl_->services.metrics->ObserveHttpRequest(metrics);
    }

    const bool more_allowed =
        ctx.request().keep_alive &&
        !impl_->draining.load(std::memory_order_acquire) &&
        impl_->requests_served < impl_->config.http.max_requests_per_connection;
    co_await WriteResponse(std::move(response), more_allowed, head_only);
    if (!more_allowed) break;
  }

  boost::system::error_code ec;
  impl_->stream.expires_never();
  impl_->stream.socket().shutdown(net::socket_base::shutdown_both, ec);
  impl_->stream.socket().close(ec);
}

net::awaitable<void> HttpSession::WriteResponse(HttpResponse response,
                                                bool keep_alive,
                                                bool head_only) {
  if (response.headers.count(http::field::server) == 0) {
    response.headers.set(http::field::server, "inferx");
  }
  if (response.headers.count(http::field::content_type) == 0) {
    response.headers.set(http::field::content_type, "application/json");
  }
  Response beast_response = ToBeastResponse(std::move(response), keep_alive,
                                            head_only);
  boost::system::error_code ec;
  impl_->stream.expires_after(impl_->config.timeouts.stream_write);
  co_await http::async_write(impl_->stream, beast_response,
                             net::redirect_error(net::use_awaitable, ec));
  // Write failures (client gone, timeout) surface as a closed loop next
  // iteration; there is nothing recoverable to do mid-response.
  (void)ec;
}

net::awaitable<void> HttpSession::WriteEarlyError(const ApiError& error) {
  HttpResponse response = ApiErrorMapper::ToHttpResponse(error);
  co_await WriteResponse(std::move(response), /*keep_alive=*/false,
                         /*head_only=*/false);
}

}  // namespace inferx::server
