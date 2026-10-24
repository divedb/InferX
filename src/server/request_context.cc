#include "inferx/server/request_context.h"

#include <atomic>
#include <chrono>
#include <mutex>
#include <random>
#include <string>
#include <utility>

#include "inferx/server/sse_writer.h"

namespace inferx::server {

namespace {

/// \brief A UUIDv7: 48-bit Unix millisecond timestamp, version 7, random
///        remainder with the RFC 4122 variant bits.
std::string GenerateUuidV7() {
  thread_local std::mt19937_64 rng([] {
    std::random_device device;
    return (static_cast<std::uint64_t>(device()) << 32) | device();
  }());
  const auto ms = static_cast<std::uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());
  constexpr std::uint64_t kVersion7 = 0x7;
  constexpr std::uint64_t kVariantRfc4122 = 0b10;
  const std::uint64_t high = (ms << 16) | (kVersion7 << 12) | (rng() & 0xFFF);
  const std::uint64_t low =
      (kVariantRfc4122 << 62) | (rng() & 0x3FFF'FFFF'FFFF'FFFFULL);

  constexpr char kHex[] = "0123456789abcdef";
  char hex[32];
  for (int i = 0; i < 16; ++i) {
    hex[i] = kHex[(high >> (60 - 4 * i)) & 0xF];
  }
  for (int i = 0; i < 16; ++i) {
    hex[16 + i] = kHex[(low >> (60 - 4 * i)) & 0xF];
  }
  std::string text;
  text.reserve(36);
  for (int i = 0; i < 32; ++i) {
    text.push_back(hex[i]);
    if (i == 7 || i == 11 || i == 15 || i == 19) text.push_back('-');
  }
  return text;
}

bool IsValidRequestId(std::string_view value) {
  if (value.size() < 8 || value.size() > 128) return false;
  for (const char c : value) {
    const bool ok = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
                    (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' ||
                    c == '~';
    if (!ok) return false;
  }
  return true;
}

}  // namespace

std::string GenerateRequestId() { return GenerateUuidV7(); }

ClientRequestId ResolveClientRequestId(const HeaderMap& headers) {
  const auto provided = headers.find("X-Request-Id");
  if (provided != headers.end() && IsValidRequestId(provided->value())) {
    return ClientRequestId(std::string(provided->value()));
  }
  // Absent, spoofed, or malformed IDs are sanitized by replacement.
  return ClientRequestId(GenerateUuidV7());
}

struct RequestContext::Impl {
  Impl(ClientRequestId client_id_in, ConnectionId connection_id_in,
       TenantId tenant_in, HttpRequest request_in,
       boost::asio::any_io_executor executor_in,
       Clock::time_point received_at_in, Clock::time_point deadline_in,
       IStreamWriter* writer_in)
      : client_id(std::move(client_id_in)),
        connection_id(std::move(connection_id_in)),
        tenant(std::move(tenant_in)),
        request(std::move(request_in)),
        executor(std::move(executor_in)),
        received_at(received_at_in),
        deadline(deadline_in),
        writer(writer_in) {}

  ClientRequestId client_id;
  ConnectionId connection_id;
  TenantId tenant;
  HttpRequest request;
  boost::asio::any_io_executor executor;
  Clock::time_point received_at;
  Clock::time_point deadline;
  CancelToken cancel;
  IStreamWriter* writer;
  RouteParams params;
  // Shared so the thread-safe RequestCancel path can reach the handle
  // without touching session memory.
  std::shared_ptr<EngineRequestHandle> engine;
  std::atomic<RequestPhase> phase{RequestPhase::kCreated};
  std::uint64_t prompt_tokens = 0;
  std::uint64_t completion_tokens = 0;
  bool response_committed = false;
};

RequestContext::RequestContext(ClientRequestId client_id,
                               ConnectionId connection_id, TenantId tenant,
                               HttpRequest request,
                               boost::asio::any_io_executor executor,
                               Clock::time_point received_at,
                               Clock::time_point deadline,
                               IStreamWriter* writer)
    : impl_(std::make_unique<Impl>(std::move(client_id),
                                   std::move(connection_id), std::move(tenant),
                                   std::move(request), std::move(executor),
                                   received_at, deadline, writer)) {}

RequestContext::~RequestContext() = default;

ClientRequestId RequestContext::client_request_id() const noexcept {
  return impl_->client_id;
}
ConnectionId RequestContext::connection_id() const noexcept {
  return impl_->connection_id;
}
TenantId RequestContext::tenant() const noexcept { return impl_->tenant; }
const HttpRequest& RequestContext::request() const noexcept {
  return impl_->request;
}
HttpRequest& RequestContext::request() noexcept { return impl_->request; }

boost::asio::any_io_executor RequestContext::executor() const noexcept {
  return impl_->executor;
}
Clock::time_point RequestContext::deadline() const noexcept {
  return impl_->deadline;
}
bool RequestContext::expired() const noexcept {
  return Clock::now() >= impl_->deadline;
}

CancelToken RequestContext::cancel_token() const noexcept {
  return impl_->cancel;
}

void RequestContext::RequestCancel(CancelReason reason) noexcept {
  impl_->cancel.Request(reason);
  if (auto engine = impl_->engine) {
    engine->Cancel(reason);
  }
}

CancelReason RequestContext::cancel_reason() const noexcept {
  return impl_->cancel.reason();
}
bool RequestContext::cancelled() const noexcept {
  return impl_->cancel.requested();
}

IStreamWriter& RequestContext::stream() noexcept { return *impl_->writer; }
bool RequestContext::response_committed() const noexcept {
  return impl_->response_committed;
}
void RequestContext::set_response_committed() noexcept {
  impl_->response_committed = true;
}

void RequestContext::set_route_params(RouteParams params) {
  impl_->params = std::move(params);
}
const RouteParams& RequestContext::route_params() const noexcept {
  return impl_->params;
}

void RequestContext::set_engine_handle(EngineRequestHandle handle) {
  impl_->engine = std::make_shared<EngineRequestHandle>(std::move(handle));
}
EngineRequestHandle* RequestContext::engine_handle() noexcept {
  return impl_->engine ? impl_->engine.get() : nullptr;
}
std::optional<EngineRequestId> RequestContext::engine_request_id() const noexcept {
  return impl_->engine ? std::optional<EngineRequestId>(impl_->engine->id())
                       : std::nullopt;
}

void RequestContext::set_usage_prompt_tokens(std::uint64_t value) noexcept {
  impl_->prompt_tokens = value;
}
void RequestContext::set_usage_completion_tokens(std::uint64_t value) noexcept {
  impl_->completion_tokens = value;
}
std::uint64_t RequestContext::prompt_tokens() const noexcept {
  return impl_->prompt_tokens;
}
std::uint64_t RequestContext::completion_tokens() const noexcept {
  return impl_->completion_tokens;
}

bool RequestContext::finalize(RequestPhase terminal) noexcept {
  // Terminal states are exactly-once: once one lands, later finals are
  // refused. Non-terminal phases cannot overwrite a terminal because only
  // finalize writes terminal values and the check+store pair runs on the
  // session strand.
  const RequestPhase current = impl_->phase.load(std::memory_order_acquire);
  if (current == RequestPhase::kCompleted ||
      current == RequestPhase::kCancelled ||
      current == RequestPhase::kFailed) {
    return false;
  }
  impl_->phase.store(terminal, std::memory_order_release);
  return true;
}

RequestPhase RequestContext::phase() const noexcept {
  return impl_->phase.load(std::memory_order_acquire);
}

}  // namespace inferx::server
