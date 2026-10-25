#ifndef INFERX_ENGINE_ENGINE_CLIENT_H_
#define INFERX_ENGINE_ENGINE_CLIENT_H_

#include <atomic>
#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/engine/request_types.h"
#include "inferx/sampling/sampling_params.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

/// \brief Why a request was cancelled: the intent side of cancellation
///        (FinishReason in TerminalEvent is the outcome side).
///
/// Passed to GenerationHandle::Cancel by whoever observes the trigger —
/// mostly the HTTP frontend, while kSlowClient's overflow path and
/// kInternal originate in the engine adapter. At most one reason is
/// recorded per request (the first wins); it surfaces in metrics labels
/// and logs, while the engine independently reports the terminal
/// FinishReason, typically kAborted.
enum class CancelReason : std::uint8_t {
  /// The transport connection was lost: the frontend's disconnect watcher
  /// saw EOF mid-generation, or a response write failed. The client is
  /// gone, so nothing is sent; the cancel stops the wasted generation.
  kClientDisconnect,

  /// The request's total time budget expired. The frontend answers 504 —
  /// or an in-stream error event once response headers are committed.
  kDeadline,

  /// No first token within the queueing budget: the request sat through
  /// queueing and prefill without producing anything. Answered with 503
  /// while the headers are still uncommitted.
  kQueueTimeout,

  /// The consumer stopped draining the stream: a socket write timed out,
  /// or the engine's bounded output buffer for this request filled
  /// (OnTokens returned kOverflow) and the adapter cancelled on its own.
  /// The connection is closed; a partially written chunk is never
  /// followed by an error chunk.
  kSlowClient,

  /// A stop string matched in the decoded text. Unusual among these: the
  /// response still completes successfully — truncated text, finish
  /// reason "stop" — the cancel just releases the engine sequence early.
  kStopString,

  /// The server is draining after a shutdown signal and the drain
  /// deadline arrived with this request still running. Clients see 503
  /// or an in-stream shutting-down error.
  kShutdown,

  /// The decoded output crossed its size budget mid-generation; the
  /// stream ends with an error event.
  kOutputLimit,

  /// The engine adapter detected a contract violation it cannot route
  /// around (a duplicate terminal, for example). The response fails with
  /// an internal error; details stay in the logs.
  kInternal,
};

/// \brief The per-request cancel rendezvous between the two parties.
///
/// Created by the engine at Submit: one copy stays in the engine's request
/// record and is polled at step boundaries, the other is wrapped in the
/// GenerationHandle handed to the caller. One packed atomic — 0 for "not
/// requested", otherwise `static_cast<uint16_t>(reason) + 1` — makes
/// first-reason-wins exact without locks or allocation.
class CancelControl {
 public:
  CancelControl() = default;
  CancelControl(const CancelControl&) = delete;
  CancelControl& operator=(const CancelControl&) = delete;

  /// \brief Idempotent; the first reason wins. Thread-safe, noexcept.
  void RequestCancel(CancelReason reason) noexcept {
    const std::uint16_t packed =
        static_cast<std::uint16_t>(reason) + 1;  // 0 reserved for "none".
    std::uint16_t expected = 0;
    packed_.compare_exchange_strong(expected, packed, std::memory_order_acq_rel,
                                    std::memory_order_relaxed);
  }

  /// \brief True once a cancel was requested.
  [[nodiscard]] bool Requested() const noexcept {
    return packed_.load(std::memory_order_acquire) != 0;
  }

  /// \brief The winning reason; valid only once Requested() is true.
  [[nodiscard]] CancelReason Reason() const noexcept {
    const std::uint16_t packed = packed_.load(std::memory_order_relaxed);
    return packed == 0 ? CancelReason::kInternal : static_cast<CancelReason>(packed - 1);
  }

 private:
  std::atomic<std::uint16_t> packed_{0};
};

/// \brief One fully normalized generation request, passed by value; the
///        engine owns its copy after Submit.
struct GenerationRequest {
  /// The prompt as token ids; tokenization is the caller's job.
  std::vector<TokenId> prompt_tokens;
  /// Sampling parameters; max_tokens is the generation budget, already
  /// clamped to the model context by the caller.
  sampling::SamplingParams sampling;
  /// Monotonic absolute deadline; advisory for the engine, enforced by the
  /// caller. Never a wall clock.
  std::chrono::steady_clock::time_point deadline{};
};

/// \brief The single terminal event of a request.
///
/// `reason` is the scheduler's FinishReason, passed through verbatim.
/// By the time OnTerminal runs, the engine has already released the
/// request's scheduler state and KV blocks.
struct TerminalEvent {
  /// kStopped and kLengthCapped are successes, kAborted a cancellation
  /// (client, deadline, shutdown, or engine-initiated), kError a failure.
  FinishReason reason = FinishReason::kStopped;
  /// Authoritative completion-token count, including the stop token, stop
  /// string overshoot, and tokens produced after a racing cancel.
  std::uint32_t completion_tokens = 0;
  /// Sanitized failure detail; valid iff kError. For logs and error
  /// mapping, never shown to end users verbatim.
  Status failure;
};

/// \brief The engine's per-request output surface, handed to the engine as
///        shared_ptr by Submit.
///
/// Implementations must be safe to call from engine threads and must not
/// depend on any executor: the consumer they serve may already be gone.
class EventSink {
 public:
  /// Per-request flow control for OnTokens.
  enum class Ack : std::uint8_t {
    kOk,        ///< Batch buffered; keep producing.
    kOverflow,  ///< Batch rejected (bounded buffer full); the engine cancels.
    kDetached,  ///< Consumer gone; drop tokens; the terminal is still owed.
  };

  virtual ~EventSink() = default;

  /// \brief Delivers one step of generated tokens.
  ///
  /// Called on an engine thread after the step that produced `ids`, with
  /// ids already copied out of executor workspace. Calls for one request
  /// are serialized. Must not block: the engine never waits on consumer
  /// progress.
  ///
  /// \param ids Token ids owned by the caller; the sink copies them.
  /// \return    Flow control for this request.
  virtual Ack OnTokens(std::span<const TokenId> ids) = 0;

  /// \brief Ends the request; called exactly once per accepted request,
  ///        after which no further sink calls occur — even when the
  ///        consumer detached earlier. Must not throw.
  ///
  /// \param event Why and how the request ended, with the authoritative
  ///              completion-token count.
  virtual void OnTerminal(TerminalEvent event) noexcept = 0;
};

/// \brief Move-only handle to an accepted request; the caller's only lever
///        on it.
///
/// Default-constructed handles are "not accepted" and compare false.
class GenerationHandle {
 public:
  /// \brief For EngineClient implementations; caller-side code receives
  ///        handles from Submit and never constructs them.
  GenerationHandle(std::shared_ptr<CancelControl> control, RequestId request_id)
      : control_(std::move(control)), id_(request_id) {}

  GenerationHandle() = default;
  GenerationHandle(GenerationHandle&&) = default;
  GenerationHandle& operator=(GenerationHandle&&) = default;
  GenerationHandle(const GenerationHandle&) = delete;
  GenerationHandle& operator=(const GenerationHandle&) = delete;

  explicit operator bool() const noexcept { return control_ != nullptr; }

  /// \brief Requests cancellation.
  ///
  /// Idempotent, thread-safe, and allocation-free: the engine observes the
  /// request at its next step boundary and then delivers a kAborted
  /// terminal, unless the request already finished. A no-op after the
  /// terminal or engine shutdown.
  ///
  /// \param reason Why the request is being cancelled; first reason wins.
  void Cancel(CancelReason reason) const noexcept {
    if (control_ != nullptr) control_->RequestCancel(reason);
  }

  /// \brief The engine-assigned id of the accepted request.
  [[nodiscard]] RequestId request_id() const noexcept { return id_; }

 private:
  std::shared_ptr<CancelControl> control_;
  RequestId id_ = 0;
};

/// \brief The asynchronous interface between the serving frontend and the
///        engine.
///
/// The engine runs its scheduler and model execution on engine-owned
/// threads; this interface is the only supported way to submit, cancel,
/// and observe generation. It is data-shaped only: intake control (when to
/// stop submitting) is the frontend's job, teardown is Shutdown's, and
/// liveness is observed through the event stream — a fatal engine failure
/// delivers a kError terminal to every live request and rejects further
/// Submits with kUnavailable. There is deliberately no polled health or
/// stats surface. All methods are thread-safe. Submit never blocks on
/// engine progress; Shutdown blocks only to cancel the remainder and join
/// the threads.
class EngineClient {
 public:
  virtual ~EngineClient() = default;

  /// \brief Submits a request without blocking.
  ///
  /// On OK the engine owns `request` and the retained `sink` and will
  /// deliver exactly one terminal through the sink; sink calls may even
  /// precede this return. On error the engine retains nothing and makes no
  /// sink calls. Rejections use a closed set: kInvalidArgument (malformed
  /// request), kResourceExhausted (submission inbox full), kUnavailable
  /// (engine failed or shutting down), kOutOfRange (prompt too long for the
  /// model context).
  ///
  /// \param request The normalized request; moved into the engine.
  /// \param sink    The per-request event sink; retained until terminal.
  /// \return        The cancellation handle, or a rejection status.
  virtual StatusOr<GenerationHandle> Submit(GenerationRequest request,
                                            std::shared_ptr<EventSink> sink) = 0;

  /// \brief Cancels every remaining request, waits for their terminals,
  ///        and joins the engine threads.
  ///
  /// From its first instruction this stops intake: Submit rejects with
  /// kUnavailable.
  /// OK means every accepted request received its terminal and all engine
  /// resources are released. On kDeadlineExceeded the engine must NOT be
  /// destroyed: threads are still live, and the process supervisor is
  /// expected to terminate the process.
  ///
  /// \param deadline How long to wait for the engine threads to exit.
  /// \return         OK, or kDeadlineExceeded on timeout.
  virtual Status Shutdown(std::chrono::milliseconds deadline) = 0;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_ENGINE_CLIENT_H_
