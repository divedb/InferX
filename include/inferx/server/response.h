/// \brief Response and error types for the serving layer
/// (docs/tokenizer_design.md §7.2).
///
/// TokenDelta.tokens is the canonical incremental token stream and .text is
/// the decoder's text suffix; neither ever repeats the full prefix. The
/// sink contract is exactly-once Close with a reserved terminal event.
#ifndef INFERX_SERVER_RESPONSE_H_
#define INFERX_SERVER_RESPONSE_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/engine/request.h"
#include "inferx/server/request_model.h"

namespace inferx {

enum class ErrorCategory : std::uint8_t {
  kInvalidInput,
  kNotFound,
  kConflict,
  kOverloaded,
  kCancelled,
  kDeadlineExceeded,
  kTokenizerFailure,
  kSchedulerRejected,
  kTransportFailure,
  kSlowConsumer,
  kShuttingDown,
  kProtocolError,
  kInternal,
};

struct RequestError {
  ErrorCategory category = ErrorCategory::kInternal;
  std::string code;     ///< Stable machine-readable code, e.g. "context_length_exceeded".
  std::string message;  ///< Safe for the client; must not contain prompt/media bytes.
  std::string param;    ///< Offending field, when applicable.
  bool retryable = false;
};

struct Usage {
  int prompt_tokens = 0;
  int completion_tokens = 0;
  int cached_tokens = 0;
};

struct TokenLogprob {
  TokenId token_id = 0;
  float logprob = 0.0F;
};

struct TokenDelta {
  std::uint64_t sequence = 0;
  std::uint32_t first_token_index = 0;
  std::vector<TokenId> tokens;         ///< Incremental, never the full prefix.
  std::string text;                    ///< Incremental text suffix; may be empty.
  std::vector<TokenLogprob> logprobs;  ///< P1; empty otherwise.
};

struct Finished {
  FinishReason reason = FinishReason::kStopped;
  std::optional<StopReason> stop;
  Usage usage;
};

struct Failed {
  RequestError error;
};

using TerminalEvent = std::variant<Finished, Failed>;

/// \brief Exactly-once terminal delivery boundary (§6.3).
///
/// - Emit is non-blocking and may return ResourceExhausted when the buffer
///   is full; the manager then elects kSlowConsumer and aborts.
/// - Close is non-blocking and must reserve/retain the terminal event even
///   if deltas have to be dropped; after Close, Emit returns
///   FailedPrecondition.
/// - A sink must not call back into the manager on the manager's stack.
class ResponseSink {
 public:
  virtual ~ResponseSink() = default;
  virtual Status Emit(TokenDelta event) = 0;
  virtual Status Close(TerminalEvent terminal) = 0;
};

struct FinalResult {
  RequestId id = 0;
  std::string text;
  std::vector<TokenId> tokens;
  FinishReason reason = FinishReason::kStopped;
  std::optional<StopReason> stop;
  Usage usage;
  std::vector<TokenLogprob> logprobs;  ///< P1
};

}  // namespace inferx

#endif  // INFERX_SERVER_RESPONSE_H_
