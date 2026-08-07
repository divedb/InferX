#pragma once

#include <chrono>

#include "inferx/server/request/request_context.h"

namespace inferx::server::request {

/// Generates a UUIDv7 request identifier with the externally visible `req_`
/// prefix. The timestamp overload exists for deterministic clock testing.
RequestId GenerateRequestId();
RequestId GenerateRequestId(std::chrono::system_clock::time_point now);

}  // namespace inferx::server::request
