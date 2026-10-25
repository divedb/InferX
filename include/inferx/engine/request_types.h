#ifndef INFERX_ENGINE_REQUEST_TYPES_H_
#define INFERX_ENGINE_REQUEST_TYPES_H_

#include <cstdint>
#include <string>
#include <utility>
#include <variant>

#include "inferx/tokenizer/id.h"

namespace inferx {

/// \brief Unique identifier for a request.
///
/// Shared by the scheduler's Request, the serving boundary
/// (engine/engine_client.h), and their implementations; one alias so the
/// id spaces cannot drift apart.
using RequestId = std::uint64_t;

/// \brief Lifecycle state of a request.
///
/// Legal transitions:
///
///   kWaiting        -> kRunning   | kFinished
///   kRunning        -> kPreempted | kFinished
///   kPreempted      -> kWaiting   | kFinished
///   kFinished       -> (terminal)
///
/// Any non-terminal state may move to kFinished (e.g. client abort).
enum class RequestStatus : std::uint8_t {
  kWaiting = 0,  ///< Queued and eligible for scheduling.
  kRunning,      ///< Admitted by the scheduler and currently active.
  kPreempted,    ///< Preempted; KV resources released, awaiting rescheduling.
  kFinished,     ///< Terminal state. See FinishReason for the cause.
};

/// \brief Represents the reason a request finished, if its status is kFinished.
///
/// This is the one finish vocabulary: the scheduler records it on Request,
/// and the serving boundary reports it verbatim in TerminalEvent
/// (kStopped/kLengthCapped are successes, kAborted a cancellation, kError a
/// failure).
enum class FinishReason : std::uint8_t {
  kStopped = 0,   ///< Generation stopped normally (EOS, stop token/string, etc.).
  kLengthCapped,  ///< Reached the configured generation/token limit.
  kAborted,       ///< Explicitly cancelled or aborted by the caller.
  kError,         ///< Terminated because of an execution/runtime error.
};

/// \brief Which stop condition fired when FinishReason is kStopped.
///
/// monostate: none/EOS; TokenId: a stop token id; std::string: a stop string.
using StopReason = std::variant<std::monostate, TokenId, std::string>;

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_TYPES_H_
