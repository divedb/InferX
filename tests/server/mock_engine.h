/// \file
/// \brief Scriptable EngineClient test double (docs/server_design.md §13,
///        §16.1). Runs on its own thread to reproduce real cross-thread
///        timing: sink calls arrive from the engine thread, Cancel is a
///        foreign-thread atomic, and terminals are delivered at step
///        boundaries exactly as the real adapter must (E1–E6).

#ifndef INFERX_TESTS_SERVER_MOCK_ENGINE_H_
#define INFERX_TESTS_SERVER_MOCK_ENGINE_H_

#include <array>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <memory>
#include <mutex>
#include <optional>
#include <thread>
#include <vector>

#include "inferx/engine/engine_client.h"

namespace inferx::server {

/// \brief MockEngine: one scripted behavior for all requests, a per-request
///        snapshot of that script, and test-only controls. Contract
///        conformance is asserted by mock_engine_test.cc; server tests use
///        it as the engine behind the HTTP layer.
class MockEngine final : public EngineClient {
 public:
  /// \brief What the engine does for one request. Snapshotted per request
  ///        at Submit.
  struct Script {
    /// Token batches emitted, one per step, in order.
    std::vector<std::vector<TokenId>> tokens_per_step;
    /// Deliver a Failed terminal for the request at this step.
    std::optional<std::uint32_t> fail_at_step;
    /// Engine-fatal failure at this step: every live request gets Failed,
    /// Further Submits reject kUnavailable (E5).
    std::optional<std::uint32_t> fatal_fail_at_step;
    /// Never emit a terminal until ReleaseWithheldTerminal or Shutdown
    /// (wedge tests, §10.7).
    bool withhold_terminal = false;
    /// Honor Cancel only after this many further steps (E4 boundary).
    std::uint32_t ignore_cancel_steps = 0;
    /// Make Submit fail with this status (E6 rejection kinds).
    std::optional<Status> reject_submit;
    /// Sleep between steps (exercises E1: Submit stays non-blocking).
    std::chrono::milliseconds step_delay{0};
    /// Finished terminal with zero tokens (§6.4 edge; an empty
    /// tokens_per_step already implies this).
    bool zero_token_finish = false;
  };

  MockEngine() = default;
  MockEngine(const MockEngine&) = delete;
  MockEngine& operator=(const MockEngine&) = delete;

  /// Stops the engine thread if it was started (tests should Shutdown
  /// explicitly; this is the never-wedged safety net).
  ~MockEngine() override;

  /// \brief Installs the script. Precondition: before Start.
  void SetScript(Script script);

  /// \brief Spawns the engine thread. Precondition: at most once.
  void Start();

  /// \brief Delivers the pending terminal of a withheld request (wedge
  ///        release). No-op for unknown or non-withheld ids.
  void ReleaseWithheldTerminal(RequestId id);

  // ---- EngineClient ----

  /// Non-blocking (E1). Rejects with: the scripted status, kUnavailable
  /// when draining/failed, kInvalidArgument for empty prompts or
  /// sampling.max_tokens == 0 (E6 closed set).
  StatusOr<GenerationHandle> Submit(
      GenerationRequest request,
      std::shared_ptr<EventSink> sink) override;

  /// Interface teardown: rejects Submits (kUnavailable, kDraining), then
  /// delivers Cancelled(kShutdown) terminals to every remaining request —
  /// withheld ones included, so a successful shutdown guarantees every
  /// accepted request received a terminal — then joins the engine thread.
  /// Idempotent. Returns kDeadlineExceeded if the thread did not exit
  /// within `deadline`.
  Status Shutdown(std::chrono::milliseconds deadline) override;

  // ---- Test observability ----

  /// Accepted minus terminals delivered.
  [[nodiscard]] std::uint32_t live_requests() const noexcept;

  /// How many distinct cancels the engine observed per reason (first
  /// reason wins; a duplicate Cancel on the same request counts once).
  [[nodiscard]] std::uint32_t cancel_calls(CancelReason reason) const noexcept;

  /// Every accepted request has received its terminal and every sink
  /// reference was dropped.
  [[nodiscard]] bool all_released() const noexcept;

 private:
  struct Record {
    RequestId id = 0;
    Script script;
    std::shared_ptr<EventSink> sink;
    std::shared_ptr<CancelControl> control;
    std::uint32_t steps_done = 0;
    bool cancel_observed = false;
    CancelReason pending_cancel_reason = CancelReason::kInternal;
    std::uint32_t steps_since_cancel = 0;
    bool withheld = false;
    /// Set by ReleaseWithheldTerminal: the pending terminal must be
    /// delivered now, not parked again because the script says withhold.
    bool terminal_released = false;
    std::uint32_t emitted_tokens = 0;
  };

  /// One engine-thread action, computed under mu_ and executed outside it.
  struct Emission {
    std::shared_ptr<EventSink> sink;
    RequestId id = 0;
    std::vector<TokenId> tokens;  // Non-empty: OnTokens first.
    std::optional<TerminalEvent> terminal;
  };

  void RunLoop();

  mutable std::mutex mu_;
  std::condition_variable cv_;  // Work arrived, stop requested, or release.
  Script script_;
  std::vector<std::unique_ptr<Record>> records_;  // Live (incl. withheld).
  std::array<std::uint32_t,
             static_cast<std::size_t>(CancelReason::kInternal) + 1>
      cancel_counts_{};
  RequestId next_id_ = 1;
  bool failed_ = false;
  bool started_ = false;
  bool stopping_ = false;
  bool thread_exited_ = false;
  std::thread thread_;
};

}  // namespace inferx::server

#endif  // INFERX_TESTS_SERVER_MOCK_ENGINE_H_
