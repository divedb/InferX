#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/tokenizer/chat_template.h"

/// The OpenAI-compatible request and response surface.
///
/// Deliberately free of engine, CUDA and socket dependencies: this layer turns
/// bytes into structs and structs into bytes, and nothing else. That is what
/// makes the awkward half of an HTTP API -- malformed JSON, missing fields,
/// hostile field types, escaping in generated text, SSE framing -- testable on
/// a machine with no GPU, which is where those bugs actually live.
namespace inferx::api {

/// \brief What a request asks the engine to generate.
///
/// Only greedy decoding is implemented, so `temperature` and `top_p` are parsed
/// and then ignored. That is a real limitation and it is recorded here rather
/// than hidden: a caller asking for temperature 1.5 gets greedy output. The
/// alternative -- rejecting every request that carries the field -- would fail
/// essentially all clients, since most send OpenAI's default of 1.0 whether or
/// not the user asked for it. Responses carry `system_fingerprint: "greedy"` so
/// the behaviour is discoverable from the wire.
struct SamplingRequest {
  int32_t max_tokens = 128;
  bool stream = false;

  /// Text sequences that end the generation. Matched against decoded output,
  /// not against token ids -- a stop string need not be a token boundary, and
  /// mapping it to ids would silently fail to fire when it is not.
  std::vector<std::string> stop;
};

/// \brief A parsed `POST /v1/chat/completions` body.
struct ChatCompletionRequest {
  std::string model;
  std::vector<tokenizer::ChatMessage> messages;
  SamplingRequest sampling;
};

/// \brief A parsed `POST /v1/completions` body.
struct CompletionRequest {
  std::string model;
  std::string prompt;
  SamplingRequest sampling;
};

/// \brief Parses a chat completion body.
///
/// Every failure is InvalidArgument with a message naming the offending field,
/// because the caller is a client author reading a 400 and nothing else.
StatusOr<ChatCompletionRequest> ParseChatCompletionRequest(
    std::string_view body);

/// \brief Parses a text completion body.
StatusOr<CompletionRequest> ParseCompletionRequest(std::string_view body);

/// \brief Token accounting returned with every response.
struct Usage {
  int32_t prompt_tokens = 0;
  int32_t completion_tokens = 0;
};

/// \brief Why generation stopped, in OpenAI's vocabulary.
///
/// Narrower than the scheduler's `FinishReason` on purpose: the API has only
/// `stop` and `length`, so a sequence retired because the block pool ran out
/// has to be reported as one of them. It is reported as `length`, since that is
/// what it is from the client's side -- output was truncated -- and inventing a
/// non-standard value would break clients that switch on this field.
enum class FinishReason { kStop, kLength };

const char* FinishReasonName(FinishReason reason);

/// \brief Builds a complete (non-streaming) chat completion response.
///
/// \param id      Response id, echoed to the client.
/// \param model   Model name to report.
/// \param content The generated text.
std::string ChatCompletionJson(std::string_view id, std::string_view model,
                               std::string_view content, FinishReason reason,
                               const Usage& usage, int64_t created);

/// \brief Builds one `chat.completion.chunk` for the streaming path.
///
/// \param content The delta since the previous chunk. May be empty, which is
///                normal -- a token can decode to no complete character.
/// \param role    Emitted in the first chunk only, as OpenAI's protocol
///                requires; empty in the rest.
/// \param reason  Non-null on the final chunk, which carries no content.
std::string ChatCompletionChunkJson(std::string_view id, std::string_view model,
                                    std::string_view role,
                                    std::string_view content,
                                    const FinishReason* reason,
                                    int64_t created);

/// \brief Builds a complete (non-streaming) text completion response.
std::string CompletionJson(std::string_view id, std::string_view model,
                           std::string_view text, FinishReason reason,
                           const Usage& usage, int64_t created);

/// \brief Builds one `text_completion` chunk for the streaming path.
std::string CompletionChunkJson(std::string_view id, std::string_view model,
                                std::string_view text,
                                const FinishReason* reason, int64_t created);

/// \brief Builds a `GET /v1/models` listing with a single entry.
std::string ModelsJson(std::string_view model, int64_t created);

/// \brief Builds an OpenAI-shaped error body.
std::string ErrorJson(std::string_view message, std::string_view type);

/// \brief Wraps `data` in one Server-Sent Events frame.
///
/// One `data:` line and a blank line. The payloads here never contain a
/// newline -- they are single-line JSON -- but the framing is written to
/// survive one anyway, because a payload that grew a newline would otherwise
/// split into two malformed events rather than failing visibly.
std::string SseFrame(std::string_view data);

/// \brief Finds the earliest occurrence of any `stop` sequence in `text`.
///
/// \return The offset to truncate at, or `npos` if none matched.
size_t FindStopSequence(std::string_view text,
                        const std::vector<std::string>& stop);

/// \brief The longest suffix of `text` that could still become a stop sequence.
///
/// Streaming cannot emit text that might turn out to be the start of a stop
/// sequence: once the bytes are on the wire they cannot be recalled, and a
/// client would see the stop string it explicitly asked to have removed. This
/// returns how many trailing bytes to withhold until more output settles the
/// question.
size_t StopSequenceHoldback(std::string_view text,
                            const std::vector<std::string>& stop);

}  // namespace inferx::api
