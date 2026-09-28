#include "inferx/dist/worker_main.h"

#include <cstddef>
#include <unordered_map>
#include <utility>
#include <vector>

#include "inferx/core/logging.h"
#include "inferx/engine/request.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model_runner.h"
#include "inferx/sampling/sampling_params.h"

namespace inferx {
namespace dist {
namespace {

/// \brief Controller-side bookkeeping for one live request: how much of the
///        prompt has KV, how many tokens were emitted, and the cap.
struct LiveRequest {
  uint32_t prompt_len = 0;
  uint32_t max_tokens = 0;
  int computed = 0;
  int emitted = 0;
  bool finished = false;
};

StatusOr<WireMessage> ReadyMessage(const WorkerOptions& options) {
  WireMessage message{};
  message.kind = MessageKind::kReady;
  message.ready.rank = options.parallel.tensor_parallel_rank;
  message.ready.world = options.parallel.tensor_parallel_size;
  return message;
}

}  // namespace

Status RunWorker(const WorkerOptions& options, const IpcChannelNames& channels) {
  const int rank = options.parallel.tensor_parallel_rank;
  const int world = options.parallel.tensor_parallel_size;
  INFERX_LOG(INFO) << "worker rank " << rank << " of " << world << " starting";

  // The model config arrives with the worker's single device assigned; the
  // parallel config names this worker's rank. Together they load exactly
  // this rank's weight shards.
  INFERX_ASSIGN_OR_RETURN(auto runner,
                          ModelRunner::Create(options.model, options.cache,
                                              options.scheduler, options.execution,
                                              options.parallel));
  const auto& checkpoint = runner->checkpoint_config();
  if (checkpoint.eos_token_id < 0) {
    return InternalError("checkpoint declares no eos token id");
  }
  Scheduler scheduler(options.scheduler, runner->kv_pool(),
                      static_cast<TokenId>(checkpoint.eos_token_id));

  INFERX_ASSIGN_OR_RETURN(auto commands, MessageChannel::Open(channels.commands));
  INFERX_ASSIGN_OR_RETURN(auto events, MessageChannel::Open(channels.events));
  INFERX_ASSIGN_OR_RETURN(auto ready, ReadyMessage(options));
  INFERX_RETURN_IF_ERROR(events.Send(ready));

  std::unordered_map<uint64_t, LiveRequest> live;
  const auto fail = [&](std::string_view what) {
    WireMessage message{};
    message.kind = MessageKind::kFailed;
    std::snprintf(message.failed.text, sizeof(message.failed.text), "%.*s",
                  static_cast<int>(what.size()), what.data());
    (void)events.Send(message);
    return InternalError(what);
  };

  while (true) {
    // Drain pending commands. With work in flight, poll briefly so engine
    // steps keep flowing; idle, block until one arrives.
    const uint32_t receive_ms = live.empty() ? 1000 : 5;
    while (true) {
      WireMessage command{};
      if (!commands.TryReceive(receive_ms, &command)) break;
      if (command.kind == MessageKind::kShutdown) {
        INFERX_LOG(INFO) << "worker rank " << rank << " shutting down";
        return OkStatus();
      }
      if (command.kind == MessageKind::kGenerate) {
        if (command.generate.num_prompt == 0 ||
            command.generate.num_prompt > kMaxIpcPromptTokens) {
          return fail("generate request prompt length out of range");
        }
        sampling::SamplingParams params;  // Greedy: the implemented sampler.
        params.temperature = 0;
        params.max_tokens = command.generate.max_tokens;
        std::vector<TokenId> prompt(command.generate.prompt,
                                    command.generate.prompt + command.generate.num_prompt);
        const Status admitted = scheduler.AddRequest(
            Request(command.generate.request_id, std::move(prompt), params));
        if (!admitted.ok()) return fail(admitted.message());
        live[command.generate.request_id] = LiveRequest{
            command.generate.num_prompt, command.generate.max_tokens, 0, 0, false};
        INFERX_VLOG(1) << "worker rank " << rank << " admitted request "
                       << command.generate.request_id;
      }
    }

    if (live.empty()) continue;

    auto plan = scheduler.Schedule();
    if (!plan.ok()) return fail(plan.status().message());
    if (plan->scheduled.empty()) continue;
    auto result = runner->Run(*plan);
    if (!result.ok()) return fail(result.status().message());
    if (const Status updated = scheduler.UpdateFromOutput(*plan, *result);
        !updated.ok()) {
      return fail(updated.message());
    }

    // Samples are real output only once the whole prompt has KV (the same
    // guard the gateway applies).
    for (std::size_t i = 0; i < plan->scheduled.size(); ++i) {
      const auto& sr = plan->scheduled[i];
      auto it = live.find(sr.request_id);
      if (it == live.end()) continue;
      LiveRequest& entry = it->second;
      entry.computed += sr.num_new_tokens;
      if (entry.computed < static_cast<int>(entry.prompt_len)) continue;
      for (const TokenId token : result->samples[i].token_ids) {
        if (entry.emitted >= static_cast<int>(entry.max_tokens)) break;
        WireMessage message{};
        message.kind = MessageKind::kToken;
        message.token.request_id = sr.request_id;
        message.token.token = token;
        INFERX_RETURN_IF_ERROR(events.Send(message));
        ++entry.emitted;
      }
    }

    while (auto finished = scheduler.PopFinished()) {
      auto it = live.find(finished->id());
      if (it == live.end()) continue;
      WireMessage message{};
      message.kind = MessageKind::kFinished;
      message.finished.request_id = finished->id();
      const FinishReason reason = finished->finish_reason();
      message.finished.finish =
          reason == FinishReason::kStopped         ? FinishKind::kStopped
          : reason == FinishReason::kLengthCapped  ? FinishKind::kLengthCapped
          : reason == FinishReason::kAborted       ? FinishKind::kAborted
                                                   : FinishKind::kError;
      INFERX_RETURN_IF_ERROR(events.Send(message));
      live.erase(it);
    }
  }
}

}  // namespace inferx::dist
}  // namespace inferx
