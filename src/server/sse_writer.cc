// ResponseWriter implementation (see sse_writer.h). Errors are returned,
// not thrown: a dead peer is routine, and the handler must cancel engine
// work rather than unwind the session coroutine.
#include "inferx/server/sse_writer.h"

#include <cstdio>

namespace inferx {

namespace {
constexpr std::string_view kSseDone = "data: [DONE]\n\n";
}

net::awaitable<bool> ResponseWriter::WriteJson(int status, const nlohmann::json& body,
                                               std::vector<Header> extra_headers) {
  if (committed_ || failed_) co_return false;
  http::response<http::string_body> res{
      static_cast<http::status>(status), 11};
  res.set(http::field::server, "inferx");
  res.set(http::field::content_type, "application/json");
  for (auto& [name, value] : extra_headers) {
    res.set(name, value);
  }
  res.keep_alive(keep_alive_);
  res.body() = body.dump();
  res.prepare_payload();

  boost::system::error_code ec;
  stream_.expires_after(write_timeout_);
  co_await http::async_write(stream_, res,
                             net::redirect_error(net::use_awaitable, ec));
  if (ec) {
    failed_ = true;
    co_return false;
  }
  committed_ = true;
  co_return true;
}

net::awaitable<bool> ResponseWriter::WriteEmpty(int status) {
  if (committed_ || failed_) co_return false;
  http::response<http::empty_body> res{static_cast<http::status>(status), 11};
  res.set(http::field::server, "inferx");
  res.keep_alive(keep_alive_);
  res.prepare_payload();  // Content-Length: 0 keeps the connection reusable.

  boost::system::error_code ec;
  stream_.expires_after(write_timeout_);
  co_await http::async_write(stream_, res,
                             net::redirect_error(net::use_awaitable, ec));
  if (ec) {
    failed_ = true;
    co_return false;
  }
  committed_ = true;
  co_return true;
}

net::awaitable<bool> ResponseWriter::BeginStream() {
  if (committed_ || failed_) co_return false;
  http::response<http::empty_body> res{http::status::ok, 11};
  res.set(http::field::server, "inferx");
  res.set(http::field::content_type, "text/event-stream; charset=utf-8");
  res.set(http::field::cache_control, "no-cache, no-transform");
  res.set("X-Accel-Buffering", "no");
  res.keep_alive(keep_alive_);
  res.chunked(true);

  http::response_serializer<http::empty_body> serializer{res};
  boost::system::error_code ec;
  stream_.expires_after(write_timeout_);
  co_await http::async_write_header(stream_, serializer,
                                    net::redirect_error(net::use_awaitable, ec));
  if (ec) {
    failed_ = true;
    co_return false;
  }
  committed_ = true;
  co_return true;
}

net::awaitable<bool> ResponseWriter::WriteRaw(std::string_view bytes) {
  if (!committed_ || failed_ || bytes.empty()) co_return !failed_;
  // The stream response is chunked; frame each payload manually (hex size,
  // bytes, CRLF) exactly like the previous SSE writer did.
  char prefix[32];
  const int digits = std::snprintf(prefix, sizeof(prefix), "%zX\r\n", bytes.size());
  std::string frame;
  frame.reserve(bytes.size() + digits + 4);
  frame.append(prefix, digits);
  frame.append(bytes.data(), bytes.size());
  frame += "\r\n";
  co_return co_await WriteAll(frame);
}

net::awaitable<bool> ResponseWriter::WriteDone() {
  if (!committed_ || failed_) co_return !failed_;
  if (done_written_) co_return true;
  const bool ok = co_await WriteRaw(kSseDone);
  if (!ok) co_return false;
  // Terminal chunk ends the chunked body; the connection stays reusable.
  done_written_ = true;
  co_return co_await WriteAll("0\r\n\r\n");
}

net::awaitable<bool> ResponseWriter::WriteAll(std::string_view bytes) {
  boost::system::error_code ec;
  stream_.expires_after(write_timeout_);
  co_await net::async_write(stream_, net::buffer(bytes.data(), bytes.size()),
                            net::redirect_error(net::use_awaitable, ec));
  if (ec) {
    failed_ = true;
    co_return false;
  }
  co_return true;
}

}  // namespace inferx
