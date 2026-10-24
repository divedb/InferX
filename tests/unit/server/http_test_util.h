// Minimal blocking HTTP/1.1 client for the Phase 1 server suites: raw
// socket writes and a small response parser, so tests assert against real
// wire bytes without a client library.
#ifndef INFERX_TESTS_UNIT_SERVER_HTTP_TEST_UTIL_H_
#define INFERX_TESTS_UNIT_SERVER_HTTP_TEST_UTIL_H_

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <poll.h>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <boost/asio.hpp>
#include <boost/system/errc.hpp>

namespace inferx::server {
namespace testing {

struct RawResponse {
  int status = 0;
  std::string reason;
  std::vector<std::pair<std::string, std::string>> headers;
  std::string body;

  const std::string* Header(const std::string& name) const {
    for (const auto& [header_name, value] : headers) {
      if (header_name == name) return &value;
    }
    return nullptr;
  }
};

class TestHttpClient {
 public:
  TestHttpClient(const std::string& host, std::uint16_t port,
                 std::chrono::milliseconds timeout = std::chrono::milliseconds{
                     5000}) {
    boost::system::error_code ec;
    socket_.connect(boost::asio::ip::tcp::endpoint(
                        boost::asio::ip::make_address(host), port),
                    ec);
    connected_ = !ec;
    if (connected_) {
      socket_.set_option(boost::asio::ip::tcp::no_delay(true), ec);
      (void)timeout;
    }
  }

  bool connected() const { return connected_; }

  void Send(const std::string& bytes) {
    if (!connected_) return;
    boost::system::error_code ec;
    boost::asio::write(socket_, boost::asio::buffer(bytes), ec);
    if (ec) connected_ = false;
  }

  void SendRequest(const std::string& method, const std::string& target,
                   const std::vector<std::pair<std::string, std::string>>& extra_headers = {},
                   const std::string& body = "") {
    std::string request = method + " " + target + " HTTP/1.1\r\n";
    request += "Host: localhost\r\n";
    for (const auto& [name, value] : extra_headers) {
      request += name + ": " + value + "\r\n";
    }
    if (!body.empty() || method == "POST" || method == "PUT") {
      request += "Content-Length: " + std::to_string(body.size()) + "\r\n";
    }
    request += "\r\n";
    request += body;
    Send(request);
  }

  /// Reads one full response (headers plus Content-Length body). HEAD
  /// responses carry headers only; pass expect_body=false for those.
  std::optional<RawResponse> ReadResponse(bool expect_body = true) {
    std::string raw;
    static constexpr const char* kHeaderEnd = "\r\n\r\n";
    while (raw.find(kHeaderEnd) == std::string::npos) {
      if (!ReadMore(raw)) return std::nullopt;
    }
    const std::size_t header_end = raw.find(kHeaderEnd) + 4;

    RawResponse response;
    std::size_t offset = 0;
    // Status line.
    const std::size_t line_end = raw.find("\r\n");
    const std::string status_line = raw.substr(0, line_end);
    offset = line_end + 2;
    const std::size_t first_space = status_line.find(' ');
    const std::size_t second_space =
        status_line.find(' ', first_space + 1);
    response.status = std::stoi(status_line.substr(
        first_space + 1, second_space - first_space - 1));
    if (second_space != std::string::npos &&
        second_space + 1 < status_line.size()) {
      response.reason = status_line.substr(second_space + 1);
    }
    // Headers.
    while (offset < header_end - 2) {
      const std::size_t next = raw.find("\r\n", offset);
      const std::string line = raw.substr(offset, next - offset);
      offset = next + 2;
      const std::size_t colon = line.find(':');
      if (colon == std::string::npos) continue;
      std::string name = line.substr(0, colon);
      std::size_t value_start = colon + 1;
      while (value_start < line.size() && line[value_start] == ' ') {
        ++value_start;
      }
      response.headers.emplace_back(std::move(name),
                                    line.substr(value_start));
    }
    std::size_t content_length = 0;
    bool has_content_length = false;
    for (const auto& [name, value] : response.headers) {
      if (name == "Content-Length") {
        content_length = static_cast<std::size_t>(std::stoull(value));
        has_content_length = true;
      }
    }
    // Body.
    if (expect_body) {
      while (raw.size() < header_end + content_length) {
        if (!ReadMore(raw)) return std::nullopt;
      }
    }
    response.body = raw.substr(header_end, content_length);
    (void)has_content_length;
    if (getenv("INFERX_TEST_DUMP_RAW")) {
      fprintf(stderr, "RAW[%zu]:", raw.size());
      for (const char c : raw) {
        if (c == '\r') fprintf(stderr, "<CR>");
        else if (c == '\n') fprintf(stderr, "<LF>");
        else if ((unsigned char)c < 0x20 || (unsigned char)c >= 0x7f)
          fprintf(stderr, "<%02X>", (unsigned char)c);
        else fputc(c, stderr);
      }
      fprintf(stderr, "\n");
    }
    return response;
  }

  /// Reads until EOF or the deadline; true when the server closes (or the
  /// deadline passes without data, treated as still-open = false).
  bool WaitForClose(std::chrono::milliseconds timeout =
                        std::chrono::milliseconds(3000)) {
    const auto deadline = std::chrono::steady_clock::now() + timeout;
    std::string sink;
    char buffer[512];
    for (;;) {
      const int remaining = static_cast<int>(
          std::chrono::duration_cast<std::chrono::milliseconds>(
              deadline - std::chrono::steady_clock::now())
              .count());
      if (remaining <= 0) return false;
      pollfd p{socket_.native_handle(), POLLIN, 0};
      const int rc = ::poll(&p, 1, remaining);
      if (rc < 0) return true;
      if (rc == 0) return false;  // still open at the deadline
      boost::system::error_code ec;
      const std::size_t got =
          socket_.read_some(boost::asio::buffer(buffer), ec);
      if (ec && got == 0) return true;  // eof or reset
      sink.append(buffer, got);
    }
  }

  void Close() {
    boost::system::error_code ec;
    socket_.close(ec);
    connected_ = false;
  }

 private:
  bool ReadMore(std::string& raw) {
    pollfd p{socket_.native_handle(), POLLIN, 0};
    const int rc = ::poll(&p, 1, 3000);
    if (rc <= 0) return false;  // timeout or poll error
    char buffer[4096];
    boost::system::error_code ec;
    const std::size_t got = socket_.read_some(boost::asio::buffer(buffer), ec);
    if (ec && got == 0) return false;
    raw.append(buffer, got);
    return true;
  }

  boost::asio::io_context io_;
  boost::asio::ip::tcp::socket socket_{io_};
  bool connected_ = false;
};

}  // namespace testing
}  // namespace inferx::server

#endif  // INFERX_TESTS_UNIT_SERVER_HTTP_TEST_UTIL_H_
