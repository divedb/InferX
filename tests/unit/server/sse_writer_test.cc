// SseWriter unit tests (docs §7.9, §13.4): one loopback TCP pair per test —
// the writer drives the server side, a raw reader drains the client side —
// so exact wire bytes, the committed/abort state machine, and transport
// failures are verified against real socket semantics.
#include <chrono>
#include <memory>
#include <poll.h>
#include <string>
#include <thread>
#include <utility>

#include <gtest/gtest.h>
#include <boost/asio.hpp>
#include <boost/beast.hpp>

#include "inferx/server/sse_writer.h"

namespace inferx::server {
namespace {

namespace net = boost::asio;
namespace beast = boost::beast;

using boost::asio::ip::tcp;

// Blocking reader on the peer side of the loopback pair.
class Peer {
 public:
  explicit Peer(tcp::socket socket) : socket_(std::move(socket)) {}

  /// Bounded read: asio's synchronous read path polls internally and
  /// ignores SO_RCVTIMEO, so the deadline is enforced with ::poll instead.
  std::string Read(std::size_t count, int timeout_ms = 500) {
    std::string out(count, '\0');
    boost::system::error_code ec;
    std::size_t total = 0;
    while (total < count) {
      pollfd p{socket_.native_handle(), POLLIN, 0};
      const int rc = ::poll(&p, 1, timeout_ms);
      if (rc <= 0) break;  // timeout or poll error
      const std::size_t got = socket_.read_some(
          net::buffer(out.data() + total, count - total), ec);
      if (ec && got == 0) break;  // eof or reset
      total += got;
    }
    out.resize(total);
    return out;
  }

  /// Reads whatever arrives within the timeout window (header + frames).
  std::string DrainAll(int timeout_ms = 300) {
    std::string all = Read(1 << 16, timeout_ms);
    return all;
  }

  void Close() {
    boost::system::error_code ec;
    socket_.close(ec);
  }

 private:
  tcp::socket socket_;
};

/// Returns the bytes after the header terminator, or "" when absent.
std::string BodyAfterHeader(const std::string& raw) {
  const std::size_t header_end = raw.find("\r\n\r\n");
  if (header_end == std::string::npos) return "";
  return raw.substr(header_end + 4);
}

/// Decodes a chunked body (RFC 9112 §7.1); returns "" when the framing is
/// malformed or the zero-length chunk is missing.
std::string Dechunk(std::string_view body) {
  std::string out;
  for (;;) {
    const std::size_t crlf = body.find("\r\n");
    if (crlf == std::string_view::npos) return "";
    const std::string size_line(body.substr(0, crlf));
    if (size_line.empty() ||
        size_line.find_first_not_of("0123456789abcdefABCDEF") !=
            std::string::npos) {
      return "";
    }
    const std::size_t size = std::stoull(size_line, nullptr, 16);
    body.remove_prefix(crlf + 2);
    if (size == 0) {
      return body == "\r\n" ? out : "";
    }
    if (body.size() < size + 2) return "";
    out.append(body.substr(0, size));
    body.remove_prefix(size);
    if (body.substr(0, 2) != "\r\n") return "";
    body.remove_prefix(2);
  }
}

struct Pair {
  tcp::socket client;
  std::unique_ptr<beast::tcp_stream> server;
};

Pair MakePair(net::io_context& io) {
  tcp::acceptor acceptor(io, tcp::endpoint(tcp::v4(), 0));
  tcp::socket client(io);
  client.connect(acceptor.local_endpoint());
  tcp::socket server_side = acceptor.accept();
  return Pair{std::move(client),
              std::make_unique<beast::tcp_stream>(std::move(server_side))};
}

class SseWriterTest : public ::testing::Test {
 protected:
  void TearDown() override {
    if (thread_.joinable()) thread_.join();
  }

  // Runs one writer coroutine on the io_context and returns its result.
  // Each call needs a fresh run(): the previous run() returned once its
  // work drained, so the finished thread is joined and the context
  // restarted before spawning.
  template <class Fn>
  auto RunOnIo(net::io_context& io, Fn&& fn) {
    if (thread_.joinable()) thread_.join();
    if (io.stopped()) io.restart();
    auto future = net::co_spawn(io.get_executor(), fn(), net::use_future);
    thread_ = std::thread([&io] { io.run(); });
    return future.get();
  }

  Result<void> Begin(SseWriter& writer, net::io_context& io) {
    return RunOnIo(io, [&]() -> net::awaitable<Result<void>> {
      HeaderMap headers;
      headers.set("Content-Type", "text/event-stream");
      headers.set("X-Accel-Buffering", "no");
      co_return co_await writer.Begin(200, std::move(headers));
    });
  }

  Result<void> WriteEvent(SseWriter& writer, net::io_context& io,
                          const std::string& json) {
    return RunOnIo(io, [&]() -> net::awaitable<Result<void>> {
      co_return co_await writer.WriteEvent(json);
    });
  }

  std::thread thread_;
};

TEST_F(SseWriterTest, BeginWritesHeaderBytes) {
  net::io_context io;
  Pair pair = MakePair(io);
  Peer peer(std::move(pair.client));
  SseWriter writer(io.get_executor(), *pair.server,
                   std::chrono::milliseconds(2000));
  ASSERT_TRUE(Begin(writer, io).ok());

  const std::string head = peer.Read(1024);
  EXPECT_NE(head.find("HTTP/1.1 200 OK\r\n"), std::string::npos);
  EXPECT_NE(head.find("text/event-stream"), std::string::npos);
  EXPECT_NE(head.find("X-Accel-Buffering: no"), std::string::npos);
  EXPECT_NE(head.find("Transfer-Encoding: chunked"), std::string::npos);
  EXPECT_NE(head.find("\r\n\r\n"), std::string::npos);
  EXPECT_TRUE(writer.committed());
}

TEST_F(SseWriterTest, EventCommentDoneFramingIsByteExact) {
  net::io_context io;
  Pair pair = MakePair(io);
  Peer peer(std::move(pair.client));
  SseWriter writer(io.get_executor(), *pair.server,
                   std::chrono::milliseconds(2000));
  ASSERT_TRUE(Begin(writer, io).ok());
  ASSERT_TRUE(WriteEvent(writer, io, R"({"a":1})").ok());
  ASSERT_TRUE(RunOnIo(io, [&]() -> net::awaitable<Result<void>> {
               co_return co_await writer.WriteComment("keep-alive");
             }).ok());
  ASSERT_TRUE(RunOnIo(io, [&]() -> net::awaitable<Result<void>> {
               co_return co_await writer.WriteDone();
             }).ok());

  // The dechunked body must be byte-exact; Dechunk also proves the chunk
  // framing itself (size lines, CRLFs, terminal zero chunk) is well formed.
  const std::string decoded = Dechunk(BodyAfterHeader(peer.DrainAll()));
  EXPECT_EQ(decoded,
            "data: {\"a\":1}\n\n"
            ": keep-alive\n\n"
            "data: [DONE]\n\n");
}

TEST_F(SseWriterTest, MultiByteUtf8SurvivesSplitReads) {
  net::io_context io;
  Pair pair = MakePair(io);
  Peer peer(std::move(pair.client));
  SseWriter writer(io.get_executor(), *pair.server,
                   std::chrono::milliseconds(2000));
  ASSERT_TRUE(Begin(writer, io).ok());
  const std::string json = "\"emoji: \xF0\x9F\x91\x8D\"";
  ASSERT_TRUE(WriteEvent(writer, io, json).ok());
  ASSERT_TRUE(RunOnIo(io, [&]() -> net::awaitable<Result<void>> {
    co_return co_await writer.WriteDone();
  }).ok());

  const std::string expected =
      "data: " + json + "\n\n"
      "data: [DONE]\n\n";
  const std::string payload = Dechunk(BodyAfterHeader(peer.DrainAll()));
  ASSERT_EQ(payload.size(), expected.size());
  // Compare in two slices to prove byte identity across read boundaries.
  const std::string first = payload.substr(0, payload.size() - 2);
  const std::string second = payload.substr(payload.size() - 2);
  EXPECT_EQ(first + second, expected);
}

TEST_F(SseWriterTest, InvalidUtf8EventFailsWithoutWriting) {
  net::io_context io;
  Pair pair = MakePair(io);
  Peer peer(std::move(pair.client));
  SseWriter writer(io.get_executor(), *pair.server,
                   std::chrono::milliseconds(2000));
  ASSERT_TRUE(Begin(writer, io).ok());
  const std::string head = peer.Read(1024);

  const std::string bad = "\"bad\xFF\"";
  EXPECT_FALSE(WriteEvent(writer, io, bad).ok());
  // Nothing beyond the header was written.
  EXPECT_TRUE(peer.Read(1).empty());
}

TEST_F(SseWriterTest, IllegalSequencesFail) {
  net::io_context io;
  Pair pair = MakePair(io);
  Peer peer(std::move(pair.client));
  SseWriter writer(io.get_executor(), *pair.server,
                   std::chrono::milliseconds(2000));

  // WriteEvent before Begin fails.
  EXPECT_FALSE(WriteEvent(writer, io, "{}").ok());

  ASSERT_TRUE(Begin(writer, io).ok());
  // Second Begin fails.
  EXPECT_FALSE(Begin(writer, io).ok());

  writer.Abort();
  // Writes after Abort fail.
  EXPECT_FALSE(RunOnIo(io, [&]() -> net::awaitable<Result<void>> {
    co_return co_await writer.WriteDone();
  }).ok());
}

TEST_F(SseWriterTest, WritesAfterDoneFail) {
  net::io_context io;
  Pair pair = MakePair(io);
  Peer peer(std::move(pair.client));
  SseWriter writer(io.get_executor(), *pair.server,
                   std::chrono::milliseconds(2000));
  ASSERT_TRUE(Begin(writer, io).ok());
  ASSERT_TRUE(RunOnIo(io, [&]() -> net::awaitable<Result<void>> {
    co_return co_await writer.WriteDone();
  }).ok());
  // A second terminal or any event after the last chunk must fail instead
  // of appending bytes to a finished body.
  EXPECT_FALSE(RunOnIo(io, [&]() -> net::awaitable<Result<void>> {
    co_return co_await writer.WriteDone();
  }).ok());
  EXPECT_FALSE(WriteEvent(writer, io, "{}").ok());
}

TEST_F(SseWriterTest, WriteAfterPeerCloseEventuallyFails) {
  net::io_context io;
  Pair pair = MakePair(io);
  Peer peer(std::move(pair.client));
  SseWriter writer(io.get_executor(), *pair.server,
                   std::chrono::milliseconds(2000));
  ASSERT_TRUE(Begin(writer, io).ok());
  peer.Close();
  // The RST may race the first write; within a bounded number of writes the
  // transport error must surface and the writer must not hang.
  bool failed = false;
  for (int i = 0; i < 8 && !failed; ++i) {
    failed = !WriteEvent(writer, io, "{}").ok();
  }
  EXPECT_TRUE(failed);
}

}  // namespace
}  // namespace inferx::server
