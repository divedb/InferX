#ifndef INFERX_SERVER_JSON_CODEC_H_
#define INFERX_SERVER_JSON_CODEC_H_

#include <string>
#include <string_view>

#include <nlohmann/json.hpp>

#include "inferx/server/server_config.h"
#include "inferx/server/status.h"

namespace inferx::server {

/// \brief Strict JSON parsing under HTTP limits: UTF-8 validity, depth <=
///        `limits.max_json_depth`, individual string length <=
///        `limits.max_json_string_bytes`, and a bounded total node count
///        (2 * depth budget per nesting level is not assumed; the ceiling is
///        proportional to the body limit). Trailing content is rejected.
[[nodiscard]] Result<nlohmann::json> ParseJson(
    std::string_view body, const HttpLimits& limits,
    std::size_t max_body_bytes);

/// \brief Serializes with strict UTF-8 validation; input produced by
///        `ParseJson` or built internally always round-trips.
[[nodiscard]] Result<std::string> DumpJson(const nlohmann::json& value);

/// \brief True when `text` is well-formed UTF-8.
[[nodiscard]] bool IsValidUtf8(std::string_view text);

}  // namespace inferx::server

#endif  // INFERX_SERVER_JSON_CODEC_H_
