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
#include <vector>

#include "inferx/engine/request.h"
#include "inferx/server/request_model.h"
#include "inferx/sampling/sampling_params.h"
#include "inferx/tokenizer/id.h"
#include "nlohmann/json.hpp"

namespace inferx {

using Json = nlohmann::json;

/// \brief A validated `/v1/completions` request.
struct CompletionsRequest {
  std::optional<ChatPrompt> chat;
  std::string prompt;               ///< Prompt text (when token_ids is empty).
  std::vector<TokenId> token_ids;   ///< Pre-tokenized prompt; skips encoding.
  sampling::SamplingParams params;  ///< Sampling fields merged over defaults.
  bool stream = false;
  /// From `stream_options.include_usage`; when false the final usage-only
  /// SSE chunk is not emitted (vLLM/OpenAI observable behavior).
  bool include_usage = false;
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
/// sampling defaults). Known unsupported generation controls return a 400
/// unsupported_parameter error, even for null/default values. Unknown
/// extension fields are ignored for forward compatibility.
/// `stop` accepts a non-empty string or up to four non-empty strings;
/// `stop_token_ids` accepts non-negative int32 IDs (vocabulary validation
/// happens at engine admission). Null and empty arrays specify no stops
/// when server defaults are empty.
/// `expected_model` is the served model name; a mismatch is a 404.
ParseResult ParseCompletionsRequest(std::string_view body,
                                    const sampling::SamplingParams& defaults,
                                    const std::string& expected_model);

ParseResult ParseChatRequest(std::string_view body,
                             const sampling::SamplingParams& defaults,
                             const std::string& expected_model);

/// \brief A client-facing request id: `prefix` ("cmpl-", "chatcmpl-") plus
/// 24 random hex characters, matching the OpenAI id shape vLLM emits.
std::string NewRequestId(std::string_view prefix);

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
                            int completion_tokens, bool chat = false);

/// \brief Assembles one `data: {...}\n\n` SSE event carrying a text delta.
/// Manual framing with JSON-escaped strings keeps the hot path tight.
std::string MakeSseChunk(const std::string& id, const std::string& model,
                         std::int64_t created, std::string_view delta,
                         std::optional<std::string_view> finish_reason, bool chat = false,
                         bool initial_role = false);

/// \brief The final usage-only event (`choices: []`, OpenAI include_usage).
std::string MakeSseUsageChunk(const std::string& id, const std::string& model,
                              std::int64_t created, int prompt_tokens,
                              int completion_tokens, bool chat = false);

/// \brief The SSE terminator line.
inline constexpr std::string_view kSseDone = "data: [DONE]\n\n";

}  // namespace inferx

#endif  // INFERX_SERVER_API_H_
