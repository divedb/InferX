/// \brief Normalized request model for the serving layer
/// (docs/tokenizer_design.md §7.1).
///
/// Protocol-specific JSON is parsed in the API adapter into these types; no
/// JSON crosses into the manager or prompt workers. Media position in the
/// prompt is part of the request contract, hence the ordered
/// ContentPart variant rather than {text, vector<media>}.
#ifndef INFERX_SERVER_REQUEST_MODEL_H_
#define INFERX_SERVER_REQUEST_MODEL_H_

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/engine/request.h"
#include "inferx/sampling/sampling_params.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

using Clock = std::chrono::steady_clock;
class ResponseSink;

struct BatchId {
  std::uint64_t value = 0;
};

enum class RequestKind : std::uint8_t {
  kGeneration,
  // kEmbedding is reserved for a future engine; it is not implemented.
};

enum class Modality : std::uint8_t {
  kImage,
  kAudio,
  kVideo,
};

struct MediaTiming {
  std::optional<double> start_seconds;
  std::optional<double> end_seconds;
};

/// Immutable byte storage for one raw media item. shared_ptr exists only on
/// the frontend side: the API adapter and prompt worker may need the same
/// bytes while an asynchronous prepare is in flight. It is never serialized.
struct MediaBuffer {
  std::vector<std::byte> bytes;
};

struct MediaItem {
  Modality modality = Modality::kImage;
  std::shared_ptr<const MediaBuffer> data;
  std::string mime_type;
  std::optional<int> audio_sample_rate_hz;
  std::optional<MediaTiming> timing;
};

struct TextPart {
  std::string text;
};

struct MediaPart {
  MediaItem media;
};

using ContentPart = std::variant<TextPart, MediaPart>;

/// Ordered by construction: media position in the prompt is preserved.
struct MultimodalInput {
  std::vector<ContentPart> parts;
};

enum class Role : std::uint8_t {
  kSystem,
  kUser,
  kAssistant,
  kTool,
};

struct ChatMessage {
  Role role = Role::kUser;
  std::vector<ContentPart> content;
  std::string name;
};

struct ChatTemplateOptions {
  bool enable_thinking = true;
  bool add_generation_prompt = true;
  bool continue_final_message = false;
};

struct ChatPrompt {
  std::vector<ChatMessage> messages;
  ChatTemplateOptions options;
};

struct TextPrompt {
  std::string text;
};

struct TokenIdsPrompt {
  std::vector<TokenId> ids;
};

/// Exactly one alternative is valid for a request. Chat is a prompt-rendering
/// input, not a scheduler input.
using PromptInput =
    std::variant<TextPrompt, TokenIdsPrompt, MultimodalInput, ChatPrompt>;

struct GenerateRequest {
  /// Optional caller-provided correlation/idempotency key. The internal
  /// RequestId is always allocated by the manager.
  std::optional<std::string> client_request_id;
  PromptInput input;
  sampling::SamplingParams sampling;
  bool stream = false;
  bool include_usage = true;
  std::optional<Clock::time_point> deadline;
  std::optional<std::string> lora_name;   // P2
  std::optional<std::string> session_id;  // P2
};

struct BatchItem {
  GenerateRequest request;
  std::shared_ptr<ResponseSink> sink;
};

struct BatchRequest {
  BatchId id;
  std::vector<BatchItem> items;
};

}  // namespace inferx

#endif  // INFERX_SERVER_REQUEST_MODEL_H_
