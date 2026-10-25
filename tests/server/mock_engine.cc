#include "mock_engine.h"

#include <algorithm>
#include <utility>

#include "absl/status/status.h"

namespace inferx::server {
namespace {

Status FatalFailure() { return absl::InternalError("mock engine fatal failure"); }

Status RequestFailure() {
  return absl::InternalError("mock per-request inference failure");
}

}  // namespace

MockEngine::~MockEngine() {
  if (thread_.joinable()) {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stopping_ = true;
    }
    cv_.notify_all();
    thread_.join();
  }
}

void MockEngine::SetScript(Script script) {
  std::lock_guard<std::mutex> lock(mu_);
  if (started_) {
    // Precondition violation, not an expected error (§16.0).
    std::terminate();
  }
  script_ = std::move(script);
}

void MockEngine::Start() {
  std::lock_guard<std::mutex> lock(mu_);
  if (started_) std::terminate();  // Precondition: at most once.
  started_ = true;
  thread_ = std::thread([this] { RunLoop(); });
}

void MockEngine::ReleaseWithheldTerminal(RequestId id) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    for (const std::unique_ptr<Record>& record : records_) {
      if (record->id == id && record->withheld) {
        // Returning to the terminal phase makes the loop deliver the
        // natural terminal: Cancelled when a cancel was observed, else
        // Finished.
        record->withheld = false;
        record->terminal_released = true;
        break;
      }
    }
  }
  cv_.notify_all();
}

StatusOr<GenerationHandle> MockEngine::Submit(
    GenerationRequest request, std::shared_ptr<EventSink> sink) {
  std::lock_guard<std::mutex> lock(mu_);
  if (script_.reject_submit.has_value()) {
    return *script_.reject_submit;
  }
  if (stopping_ || failed_) {
    return absl::UnavailableError("mock engine is draining or failed");
  }
  if (request.prompt_tokens.empty() || request.sampling.max_tokens == 0) {
    return absl::InvalidArgumentError(
        "mock engine: prompt_tokens and sampling.max_tokens must be non-empty");
  }

  auto record = std::make_unique<Record>();
  record->id = next_id_++;
  record->script = script_;
  record->sink = std::move(sink);
  record->control = std::make_shared<CancelControl>();
  const RequestId id = record->id;
  const std::shared_ptr<CancelControl> control = record->control;
  records_.push_back(std::move(record));
  cv_.notify_all();
  return GenerationHandle(control, id);
}

Status MockEngine::Shutdown(std::chrono::milliseconds deadline) {
  {
    std::lock_guard<std::mutex> lock(mu_);
    stopping_ = true;
  }
  cv_.notify_all();

  {
    std::unique_lock<std::mutex> lock(mu_);
    if (!cv_.wait_for(lock, deadline, [this] { return thread_exited_; })) {
      return absl::DeadlineExceededError("mock engine did not stop in time");
    }
  }
  if (thread_.joinable()) thread_.join();
  return OkStatus();
}

std::uint32_t MockEngine::live_requests() const noexcept {
  std::lock_guard<std::mutex> lock(mu_);
  return static_cast<std::uint32_t>(records_.size());
}

std::uint32_t MockEngine::cancel_calls(CancelReason reason) const noexcept {
  std::lock_guard<std::mutex> lock(mu_);
  return cancel_counts_[static_cast<std::size_t>(reason)];
}

bool MockEngine::all_released() const noexcept {
  std::lock_guard<std::mutex> lock(mu_);
  return records_.empty();
}

void MockEngine::RunLoop() {
  std::vector<Emission> emissions;
  std::vector<RequestId> finished_ids;
  std::chrono::milliseconds delay{0};

  for (;;) {
    emissions.clear();
    finished_ids.clear();

    {
      std::unique_lock<std::mutex> lock(mu_);
      cv_.wait(lock, [this] {
        if (stopping_) return true;
        for (const std::unique_ptr<Record>& record : records_) {
          if (!record->withheld) return true;
        }
        return false;
      });

      if (stopping_) {
        // A successful engine shutdown guarantees every accepted request a
        // terminal (§10.7) — withheld ones included.
        for (std::unique_ptr<Record>& record : records_) {
          Emission emission;
          emission.sink = record->sink;
          emission.id = record->id;
          emission.terminal = TerminalEvent{};
          emission.terminal->reason = FinishReason::kAborted;
          emission.terminal->completion_tokens = record->emitted_tokens;
          emissions.push_back(std::move(emission));
        }
        records_.clear();
      } else {
        bool fatal = false;
        for (std::unique_ptr<Record>& record : records_) {
          if (record->withheld) continue;  // Waiting for ReleaseWithheldTerminal.

          // Observe a foreign-thread cancel at the step boundary (E4).
          if (!record->cancel_observed && record->control->Requested()) {
            record->cancel_observed = true;
            record->pending_cancel_reason = record->control->Reason();
            record->steps_since_cancel = 0;
            ++cancel_counts_[static_cast<std::size_t>(record->pending_cancel_reason)];
          }

          const std::uint32_t step = record->steps_done;
          const std::size_t token_steps = record->script.tokens_per_step.size();
          const bool cancel_due =
              record->cancel_observed &&
              record->steps_since_cancel >= record->script.ignore_cancel_steps;
          const bool terminal_phase = step >= token_steps;

          auto make_terminal = [&](TerminalEvent event) {
            Emission emission;
            emission.sink = record->sink;
            emission.id = record->id;
            emission.terminal = std::move(event);
            emissions.push_back(std::move(emission));
            finished_ids.push_back(record->id);
          };

          if (cancel_due) {
            TerminalEvent event;
            event.reason = FinishReason::kAborted;
            event.completion_tokens = record->emitted_tokens;
            make_terminal(std::move(event));
          } else if (record->script.fatal_fail_at_step == step) {
            fatal = true;
            break;
          } else if (record->script.fail_at_step == step) {
            TerminalEvent event;
            event.reason = FinishReason::kError;
            event.failure = RequestFailure();
            event.completion_tokens = record->emitted_tokens;
            make_terminal(std::move(event));
          } else if (!terminal_phase) {
            Emission emission;
            emission.sink = record->sink;
            emission.id = record->id;
            emission.tokens = record->script.tokens_per_step[step];
            record->emitted_tokens +=
                static_cast<std::uint32_t>(emission.tokens.size());
            ++record->steps_done;
            if (record->cancel_observed) ++record->steps_since_cancel;
            emissions.push_back(std::move(emission));
          } else if (record->script.withhold_terminal && !record->terminal_released) {
            record->withheld = true;  // Park until released or shutdown.
          } else {
            TerminalEvent event;
            event.reason = FinishReason::kStopped;
            event.completion_tokens = record->emitted_tokens;
            make_terminal(std::move(event));
          }
        }

        if (fatal) {
          failed_ = true;
          for (std::unique_ptr<Record>& record : records_) {
            Emission emission;
            emission.sink = record->sink;
            emission.id = record->id;
            emission.terminal = TerminalEvent{};
            emission.terminal->reason = FinishReason::kError;
            emission.terminal->failure = FatalFailure();
            emission.terminal->completion_tokens = record->emitted_tokens;
            emissions.push_back(std::move(emission));
          }
          records_.clear();
        } else {
          for (const RequestId id : finished_ids) {
            records_.erase(std::remove_if(records_.begin(), records_.end(),
                                          [id](const std::unique_ptr<Record>& record) {
                                            return record->id == id;
                                          }),
                           records_.end());
          }
        }
        delay = script_.step_delay;
      }
    }

    // Sink calls happen on the engine thread, outside mu_ (I3 discipline).
    for (Emission& emission : emissions) {
      if (!emission.tokens.empty()) {
        emission.sink->OnTokens(emission.tokens);
      }
      if (emission.terminal.has_value()) {
        emission.sink->OnTerminal(std::move(*emission.terminal));
      }
    }

    {
      std::lock_guard<std::mutex> lock(mu_);
      if (stopping_) {
        thread_exited_ = true;
        cv_.notify_all();
        return;
      }
    }
    if (!emissions.empty() && delay > std::chrono::milliseconds::zero()) {
      std::this_thread::sleep_for(delay);
    }
  }
}

}  // namespace inferx::server
