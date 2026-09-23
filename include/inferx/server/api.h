// OpenAI-compatible API schemas for `inferx serve`: request parsing and
// validation, response builders, and SSE framing. Pure functions over
// nlohmann JSON, deliberately free of asio/beast so they stay unit-testable
// without a network or GPU. The wire contract follows server.md, which is
// grounded in vllm bench serve's request function.
#ifndef INFERX_SERVER_API_H_
#define INFERX_SERVER_API_H_

#include <cstdint>
#include <optional>
#include <string>
#include <string_view>

#include "inferx/engine/request.h"
#include "inferx/sampling/sampling_params.h"
#include "nlohmann/json.hpp"

namespace inferx::server {

using Json = nlohmann::json;

/// \brief A validated `/v1/completions` request.
struct CompletionsRequest {
  std::string prompt;               ///< Prompt text (token arrays come later).
  sampling::SamplingParams params;  ///< Sampling fields merged over defaults.
  bool stream = false;
};

/// \brief Outcome of parsing; exactly one of `request` / the error fields.
struct ParseResult {
  bool ok = false;
  CompletionsRequest request;
  int http_status = 0;         ///< 400 / 404 on failure.
  std::string message;         ///< Human-readable error detail.
  std::string param;           ///< Offending field, empty when none.
  std::string code;            ///< Stable error code, empty when none.
};

/// \brief Parses and validates a completions request body.
///
/// Fields absent from the body fall back to `defaults` (the server's
/// sampling defaults). Unknown fields are ignored for forward compatibility.
/// `expected_model` is the served model name; a mismatch is a 404.
ParseResult ParseCompletionsRequest(std::string_view body,
                                    const sampling::SamplingParams& defaults,
                                    const std::string& expected_model);

/// \brief Maps an engine finish reason onto the API spelling.
/// Only kStopped/kLengthCapped are user-visible; other reasons surface as
/// errors before a finish chunk is written.
std::string_view FinishReasonName(FinishReason reason);

/// \brief OpenAI-style error body.
Json MakeError(int status, std::string message,
               std::string param = {}, std::string code = {});

/// \brief `GET /v1/models` payload.
Json MakeModelList(const std::string& model, std::int64_t created);

/// \brief Non-streaming completion response body.
Json MakeCompletionResponse(const std::string& id, const std::string& model,
                            std::int64_t created, const std::string& text,
                            std::string_view finish_reason, int prompt_tokens,
                            int completion_tokens);

/// \brief Assembles one `data: {...}\n\n` SSE event carrying a text delta.
/// Manual framing with JSON-escaped strings keeps the hot path tight.
std::string MakeSseChunk(const std::string& id, const std::string& model,
                         std::int64_t created, std::string_view delta,
                         std::optional<std::string_view> finish_reason);

/// \brief The final usage-only event (`choices: []`, OpenAI include_usage).
std::string MakeSseUsageChunk(const std::string& id, const std::string& model,
                              std::int64_t created, int prompt_tokens,
                              int completion_tokens);

/// \brief The SSE terminator line.
inline constexpr std::string_view kSseDone = "data: [DONE]\n\n";

}  // namespace inferx::server

#endif  // INFERX_SERVER_API_H_
