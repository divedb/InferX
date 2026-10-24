#ifndef INFERX_SERVER_ENGINE_TYPES_H_
#define INFERX_SERVER_ENGINE_TYPES_H_

#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "inferx/server/status.h"

namespace inferx::server {

enum class EngineTask : std::uint8_t { kGenerate, kEmbed, kScore };
enum class ModelTask : std::uint8_t { kGenerate, kEmbed, kScore, kClassify };

// Located in the engine types header so that engine_client.h does not depend
// on request_context.h (which includes engine_client.h).
enum class CancelReason : std::uint8_t {
  kNone = 0,
  kClientDisconnected,
  kClientTimeout,
  kServerDrain,
  kAdmissionTimeout,
  kStreamOverflow,
  kInternal,
};

struct CancelState {
  std::atomic<bool> requested{false};
  std::atomic<CancelReason> reason{CancelReason::kNone};
};

/// \brief A copyable, thread-safe cancellation token. It is deliberately not
///        an asio::cancellation_slot: one slot can have only one handler,
///        while a request has several observers (engine submit, engine
///        stream, HTTP writer, timers). The asio::cancellation_signal remains
///        private to each local socket operation.
class CancelToken {
 public:
  CancelToken() : state_(std::make_shared<CancelState>()) {}

  void Request(CancelReason reason) noexcept {
    if (reason == CancelReason::kNone) return;
    state_->reason.store(reason, std::memory_order_relaxed);
    state_->requested.store(true, std::memory_order_release);
  }
  [[nodiscard]] bool requested() const noexcept {
    return state_->requested.load(std::memory_order_acquire);
  }
  [[nodiscard]] CancelReason reason() const noexcept {
    return state_->reason.load(std::memory_order_relaxed);
  }

 private:
  std::shared_ptr<CancelState> state_;
};

enum class Role : std::uint8_t {
  kSystem,
  kDeveloper,
  kUser,
  kAssistant,
  kTool,
  kFunction,
};

enum class MediaKind : std::uint8_t { kImage, kAudio, kVideo };

struct TextPart {
  std::string text;
};

struct MediaPart {
  MediaKind kind = MediaKind::kImage;
  std::string mime_type;
  // Inline bytes after base64 decode. Shared because the media subsystem may
  // retain immutable bytes for caching; it is not a zero-copy claim.
  std::shared_ptr<const std::vector<std::byte>> bytes;
  // Set only when remote media is enabled and accepted by policy.
  std::string source_uri;
  std::optional<std::uint32_t> width;
  std::optional<std::uint32_t> height;
  std::optional<std::uint32_t> duration_ms;
};

using ContentPart = std::variant<TextPart, MediaPart>;

struct ToolCall {
  std::string id;
  std::string name;
  std::string arguments_json;
};

/// \brief One chat message with ordered content parts; order is semantically
///        significant and preserved end to end.
struct ChatMessage {
  Role role = Role::kUser;
  std::vector<ContentPart> parts;
  std::optional<std::string> name;
  std::vector<ToolCall> tool_calls;
  std::optional<std::string> tool_call_id;
};

struct ChatPrompt {
  std::vector<ChatMessage> messages;
};

struct TextPrompt {
  std::string text;
};

struct TokenIdPrompt {
  std::vector<std::int32_t> token_ids;
};

struct SamplingParams {
  std::uint32_t n = 1;
  std::optional<std::int32_t> max_tokens;
  std::optional<double> temperature;
  std::optional<double> top_p;
  std::optional<double> top_k;
  std::optional<double> min_p;
  std::optional<double> presence_penalty;
  std::optional<double> frequency_penalty;
  std::optional<double> repetition_penalty;
  std::uint64_t seed = 0;
  bool has_seed = false;
  std::vector<std::string> stop_strings;
  std::vector<std::int32_t> stop_token_ids;
  bool include_stop_str_in_output = false;
  bool ignore_eos = false;
  std::optional<std::int32_t> logprobs;
  std::optional<std::int32_t> prompt_logprobs;
  bool echo = false;
  std::optional<std::string> structured_outputs_json;
  // Opaque, versioned extension object. The server does not interpret it.
  std::optional<std::string> extension_json;
};

struct RenderOptions {
  std::optional<std::string> request_chat_template;  // only if trusted
  std::optional<std::string> chat_template_kwargs_json;
  bool add_generation_prompt = true;
  bool continue_final_message = false;
  bool return_token_ids = false;
  bool return_tokens_as_token_ids = false;
};

struct TraceContext {
  std::string traceparent;  // W3C traceparent if present
  std::string tracestate;
  std::string request_id;  // client ID for logs, not engine identity
};

/// \brief Normalized generation request. Wire DTOs are converted into this
///        shape; the engine never sees HTTP structures.
struct EngineGenerateRequest {
  EngineTask task = EngineTask::kGenerate;
  std::string model;  // resolved catalog ID, not raw user alias
  std::variant<ChatPrompt, TextPrompt, TokenIdPrompt> input;
  RenderOptions render;
  SamplingParams sampling;
  TraceContext trace;
  std::int32_t priority = 0;
  bool stream = false;
  bool include_usage = false;
  bool continuous_usage = false;

  EngineGenerateRequest() = default;
  EngineGenerateRequest(const EngineGenerateRequest&) = delete;
  EngineGenerateRequest& operator=(const EngineGenerateRequest&) = delete;
  EngineGenerateRequest(EngineGenerateRequest&&) = default;
  EngineGenerateRequest& operator=(EngineGenerateRequest&&) = default;
  ~EngineGenerateRequest() = default;
};

struct EngineEmbedRequest {
  EngineTask task = EngineTask::kEmbed;
  std::string model;
  std::vector<std::string> inputs;
  std::optional<std::uint32_t> dimensions;
  std::optional<std::string> input_type;
  TraceContext trace;
  std::int32_t priority = 0;
  bool normalize = true;

  EngineEmbedRequest() = default;
  EngineEmbedRequest(const EngineEmbedRequest&) = delete;
  EngineEmbedRequest& operator=(const EngineEmbedRequest&) = delete;
  EngineEmbedRequest(EngineEmbedRequest&&) = default;
  EngineEmbedRequest& operator=(EngineEmbedRequest&&) = default;
};

enum class FinishReason : std::uint8_t {
  kStop,
  kLength,
  kAbort,
  kError,
  kRepetition,
  kContentFilter,
  kToolCalls,
};

struct Usage {
  std::uint64_t prompt_tokens = 0;
  std::uint64_t completion_tokens = 0;
  std::uint64_t total_tokens = 0;
  std::uint64_t cached_tokens = 0;
};

struct AcceptedEvent {
  std::uint64_t prompt_tokens = 0;
  std::string engine_model;
  std::string system_fingerprint;
};

struct DeltaEvent {
  std::uint32_t sequence_index = 0;
  std::string text;  // valid UTF-8, complete code points
  std::vector<std::int32_t> token_ids;
  std::optional<std::string> logprobs_json;
  std::optional<FinishReason> finish_reason;
  std::optional<std::string> stop_reason;
};

struct UsageEvent {
  Usage usage;
};

struct FinishedEvent {
  std::uint32_t sequence_index = 0;
  FinishReason reason = FinishReason::kStop;
  std::optional<std::string> stop_reason;
  Usage usage;
  std::optional<std::uint64_t> engine_ttft_us;
};

struct EngineErrorEvent {
  ApiError error;
  bool fatal = false;  // fatal => engine/readiness transition
};

/// \brief One engine event. `seq` must be strictly +1 within one request.
struct EngineGenerateEvent {
  EngineRequestId request_id;
  std::uint64_t seq = 0;
  std::variant<AcceptedEvent, DeltaEvent, UsageEvent, FinishedEvent,
               EngineErrorEvent>
      payload;

  EngineGenerateEvent() = default;
  EngineGenerateEvent(const EngineGenerateEvent&) = delete;
  EngineGenerateEvent& operator=(const EngineGenerateEvent&) = delete;
  EngineGenerateEvent(EngineGenerateEvent&&) = default;
  EngineGenerateEvent& operator=(EngineGenerateEvent&&) = default;
};

enum class SubmitDisposition : std::uint8_t {
  kEngineAccepted,                  // execution admitted; first event may be Accepted
  kTransportAcceptedAwaitingEngine  // first event MUST be Accepted or Error
};

struct EngineSubmission {
  EngineRequestId id;
  SubmitDisposition disposition = SubmitDisposition::kEngineAccepted;
  std::unique_ptr<class IEngineEventStream> events;
  Usage estimated_usage;
};

struct EmbeddingItem {
  std::uint32_t index = 0;
  std::vector<float> values;
};

struct EmbedFinishedEvent {
  Usage usage;
};

struct EngineEmbedEvent {
  EngineRequestId request_id;
  std::uint64_t seq = 0;
  std::variant<AcceptedEvent, EmbeddingItem, EmbedFinishedEvent,
               EngineErrorEvent>
      payload;

  EngineEmbedEvent() = default;
  EngineEmbedEvent(const EngineEmbedEvent&) = delete;
  EngineEmbedEvent& operator=(const EngineEmbedEvent&) = delete;
  EngineEmbedEvent(EngineEmbedEvent&&) = default;
  EngineEmbedEvent& operator=(EngineEmbedEvent&&) = default;
};

struct EngineEmbedSubmission {
  EngineRequestId id;
  SubmitDisposition disposition = SubmitDisposition::kEngineAccepted;
  std::unique_ptr<class IEngineEmbedStream> events;
  Usage estimated_usage;
};

}  // namespace inferx::server

#endif  // INFERX_SERVER_ENGINE_TYPES_H_
