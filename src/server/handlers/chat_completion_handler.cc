#include "inferx/server/handlers/chat_completion_handler.h"

#include "inferx/api/openai.h"
#include "inferx/server/api/error_mapping.h"
#include "inferx/server/request/request_id.h"
#include "inferx/server/transport/sse_writer.h"
#include "inferx/support/log.h"

namespace inferx::server::handlers {
namespace {

// The handler runs parallel choices as sequential engine requests, so every
// extra choice multiplies the request's engine time; this bounds what one
// HTTP request can demand. The parser's own cap is a sanity bound only.
constexpr int32_t kMaxParallelChoices = 8;

folly::coro::Task<void> WriteError(
    const transport::HttpRequest& request, transport::ResponseWriter& writer,
    const Status& status, std::string_view param,
    folly::CancellationToken cancellation) {
  auto error = api::MapStatus(status, param);
  transport::HttpResponse response{
      static_cast<boost::beast::http::status>(error.status), request.version()};
  response.keep_alive(request.keep_alive());
  response.set(boost::beast::http::field::content_type, "application/json");
  response.body() = error.Json({});
  response.prepare_payload();
  (void)co_await writer.WriteResponse(std::move(response), cancellation);
}

::inferx::api::FinishReason ApiFinish(
    scheduler_client::FinishReason reason) {
  return reason == scheduler_client::FinishReason::kStop
             ? ::inferx::api::FinishReason::kStop
             : ::inferx::api::FinishReason::kLength;
}

const char* FinishName(scheduler_client::FinishReason reason) {
  return reason == scheduler_client::FinishReason::kStop ? "stop" : "length";
}

/// What one choice's drained event stream amounts to.
struct ChoiceOutcome {
  std::string text;
  scheduler_client::FinishReason finish = scheduler_client::FinishReason::kNone;
  scheduler_client::Usage usage;
  std::vector<scheduler_client::TokenLogprobs> logprobs;
};

/// Turns the scheduler's id-keyed logprobs into the api layer's text-keyed
/// mirror. One DecodeTokens call covers chosen tokens and alternatives alike,
/// because each decode batch costs a tokenizer clone.
StatusOr<::inferx::api::ChoiceLogprobs> DecodeLogprobs(
    tokenization::TokenizationService* tokenization,
    const request::ModelVersion& version,
    const std::vector<scheduler_client::TokenLogprobs>& raw) {
  std::vector<tokenizer::TokenId> ids;
  for (const auto& entry : raw) {
    ids.push_back(entry.token_id);
    for (const auto& [id, logprob] : entry.top) ids.push_back(id);
  }
  INFERX_ASSIGN_OR_RETURN(std::vector<std::string> texts,
                          tokenization->DecodeTokens(version, ids));
  ::inferx::api::ChoiceLogprobs out;
  out.tokens.reserve(raw.size());
  size_t at = 0;
  for (const auto& entry : raw) {
    ::inferx::api::TokenLogprob token;
    token.token = std::move(texts[at++]);
    token.logprob = entry.logprob;
    token.top.reserve(entry.top.size());
    for (const auto& [id, logprob] : entry.top) {
      token.top.emplace_back(std::move(texts[at++]), logprob);
    }
    out.tokens.push_back(std::move(token));
  }
  return out;
}

}  // namespace

folly::coro::Task<void> ChatCompletionHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  const auto request_started = std::chrono::steady_clock::now();
  if (!context.authenticated) {
    co_await WriteError(request, response,
                        absl::UnauthenticatedError(
                            "authenticated request context is required"),
                        {}, cancellation);
    co_return;
  }
  auto parsed = ::inferx::api::ParseChatCompletionRequest(request.body());
  if (!parsed.ok()) {
    co_await WriteError(request, response, parsed.status(), {}, cancellation);
    co_return;
  }
  if (parsed->sampling.stream && parsed->sampling.n > 1) {
    co_await WriteError(request, response,
                        InvalidArgumentError(
                            "streaming with n>1 is not supported by InferX"),
                        "n", cancellation);
    co_return;
  }
  if (parsed->sampling.n > kMaxParallelChoices) {
    co_await WriteError(request, response,
                        InvalidArgumentError(
                            "n exceeds the server's parallel-choice limit of ",
                            kMaxParallelChoices),
                        "n", cancellation);
    co_return;
  }
  if (registry_ == nullptr || tokenization_ == nullptr || requests_ == nullptr) {
    co_await WriteError(request, response,
                        InternalError("generation services are not configured"),
                        {}, cancellation);
    co_return;
  }
  auto model = registry_->Resolve(parsed->model, context.tenant_id);
  if (!model.ok()) {
    co_await WriteError(request, response, model.status(), "model",
                        cancellation);
    co_return;
  }
  if (!model->supports_generation) {
    co_await WriteError(request, response,
                        UnimplementedError(
                            "model does not support generation workloads"),
                        "model", cancellation);
    co_return;
  }
  const ::inferx::server::request::ModelVersion version =
      model->id + "@" + model->version;
  auto tokenized = tokenization_->TokenizeChat(version, parsed->messages);
  if (!tokenized.ok()) {
    co_await WriteError(request, response, tokenized.status(), "messages",
                        cancellation);
    co_return;
  }
  if (model->context_limit != 0 &&
      tokenized->prompt_tokens + parsed->sampling.max_tokens >
          model->context_limit) {
    co_await WriteError(request, response,
                        InvalidArgumentError(
                            "messages and max_tokens exceed model context limit"),
                        "max_tokens", cancellation);
    co_return;
  }

  const uint64_t base_seed =
      parsed->sampling.has_seed ? parsed->sampling.seed : 0;
  const auto make_command = [&](int32_t choice) {
    scheduler_client::ScheduledRequest command;
    command.request_id = ::inferx::server::request::GenerateRequestId();
    command.tenant_id = context.tenant_id;
    command.model_version = version;
    command.tokenizer_revision = model->tokenizer_revision;
    command.workload = scheduler_client::WorkloadClass::kGeneration;
    // Copied, not moved: every choice submits the same prompt.
    command.prompt_tokens = tokenized->token_ids;
    command.sampling.max_tokens = parsed->sampling.max_tokens;
    command.sampling.temperature = parsed->sampling.temperature;
    command.sampling.top_p = parsed->sampling.top_p;
    command.sampling.top_k = parsed->sampling.top_k;
    command.sampling.min_p = parsed->sampling.min_p;
    command.sampling.presence_penalty = parsed->sampling.presence_penalty;
    command.sampling.frequency_penalty = parsed->sampling.frequency_penalty;
    command.sampling.repetition_penalty = parsed->sampling.repetition_penalty;
    // Each choice draws with a distinct seed; choice 0 keeps the request's
    // own so a former n=1 request reproduces exactly.
    command.sampling.seed =
        choice == 0 ? parsed->sampling.seed : base_seed + choice;
    command.sampling.stop = parsed->sampling.stop;
    command.sampling.stop_token_ids = parsed->sampling.stop_token_ids;
    command.sampling.ignore_eos = parsed->sampling.ignore_eos;
    command.sampling.min_tokens = parsed->sampling.min_tokens;
    command.sampling.skip_special_tokens = parsed->sampling.skip_special_tokens;
    command.sampling.include_stop_str_in_output =
        parsed->sampling.include_stop_str_in_output;
    command.sampling.want_logprobs = parsed->sampling.want_logprobs;
    command.sampling.top_logprobs = parsed->sampling.top_logprobs;
    command.deadline = std::chrono::steady_clock::now() + timeout_;
    return command;
  };

  const int64_t created = std::chrono::duration_cast<std::chrono::seconds>(
                              std::chrono::system_clock::now().time_since_epoch())
                              .count();
  const bool sampled = parsed->sampling.temperature > 0;

  auto submitted = co_await requests_->Submit(make_command(0), cancellation);
  if (!submitted.ok()) {
    co_await WriteError(request, response, submitted.status(), {}, cancellation);
    co_return;
  }
  const auto queued_at = std::chrono::steady_clock::now();

  auto events = requests_->Events(submitted->request_id, cancellation);
  scheduler_client::Usage usage;
  std::optional<std::chrono::steady_clock::time_point> first_token_at;
  transport::SseWriter sse(&response);
  if (parsed->sampling.stream) {
    Status started = co_await sse.Start(request.version(), request.keep_alive(),
                                        submitted->request_id, cancellation);
    if (!started.ok()) {
      (void)co_await requests_->Cancel(
          submitted->request_id,
          ::inferx::server::request::CancellationReason::kClientDisconnected);
      co_return;
    }
    Status role = co_await sse.Event(
        ::inferx::api::ChatCompletionChunkJson(
            submitted->request_id, parsed->model, "assistant", {}, nullptr,
            created, sampled),
        cancellation);
    if (!role.ok()) {
      (void)co_await requests_->Cancel(
          submitted->request_id,
          ::inferx::server::request::CancellationReason::kClientDisconnected);
      co_return;
    }
  }

  std::vector<ChoiceOutcome> outcomes;
  outcomes.emplace_back();
  scheduler_client::FinishReason finish = scheduler_client::FinishReason::kNone;
  while (auto next = co_await events.next()) {
    auto event = std::move(*next);
    if (!first_token_at.has_value() &&
        (!event.text_delta.empty() || !event.token_ids.empty() ||
         event.terminal)) {
      first_token_at = std::chrono::steady_clock::now();
    }
    if (!event.error.ok()) {
      if (!parsed->sampling.stream) {
        co_await WriteError(request, response, event.error, {}, cancellation);
      } else {
        auto error = api::MapStatus(event.error);
        (void)co_await sse.Event(error.Json(submitted->request_id), cancellation);
        (void)co_await sse.Finish(cancellation);
      }
      co_return;
    }
    outcomes.back().usage = event.usage;
    if (parsed->sampling.stream) {
      // A token that completed no character still carries its logprob entry,
      // so such an event must produce a chunk whenever a report is due --
      // otherwise the client's per-token accounting silently loses positions.
      if (!event.text_delta.empty() || !event.logprobs.empty()) {
        ::inferx::api::ChoiceLogprobs decoded;
        const ::inferx::api::ChoiceLogprobs* logprobs = nullptr;
        if (!event.logprobs.empty()) {
          auto converted =
              DecodeLogprobs(tokenization_, version, event.logprobs);
          if (!converted.ok()) {
            auto error = api::MapStatus(converted.status());
            (void)co_await sse.Event(error.Json(submitted->request_id),
                                     cancellation);
            (void)co_await sse.Finish(cancellation);
            (void)co_await requests_->Cancel(
                submitted->request_id,
                ::inferx::server::request::CancellationReason::kInternal);
            co_return;
          }
          decoded = std::move(*converted);
          logprobs = &decoded;
        }
        Status written = co_await sse.Event(
            ::inferx::api::ChatCompletionChunkJson(
                submitted->request_id, parsed->model, {}, event.text_delta,
                nullptr, created, sampled, logprobs),
            cancellation);
        if (!written.ok()) {
          (void)co_await requests_->Cancel(
              submitted->request_id,
              ::inferx::server::request::CancellationReason::kClientDisconnected);
          co_return;
        }
      }
    } else {
      outcomes.back().text += event.text_delta;
      for (auto& logprob : event.logprobs) {
        outcomes.back().logprobs.push_back(std::move(logprob));
      }
    }
    if (event.terminal) {
      finish = event.finish_reason;
      outcomes.back().finish = finish;
      break;
    }
  }
  if (finish == scheduler_client::FinishReason::kNone) {
    if (!parsed->sampling.stream) {
      co_await WriteError(request, response,
                          InternalError("generation stream ended without a "
                                        "terminal event"),
                          {}, cancellation);
    } else {
      (void)co_await sse.Finish(cancellation);
    }
    co_return;
  }
  usage = outcomes.front().usage;

  // Remaining choices run after the first finishes. Sequential is deliberate:
  // events buffer server-side, and one request must not burst-fill the
  // engine's admission budget just because it asked for eight variants.
  for (int32_t choice = 1; choice < parsed->sampling.n; ++choice) {
    auto choice_submitted =
        co_await requests_->Submit(make_command(choice), cancellation);
    if (!choice_submitted.ok()) {
      co_await WriteError(request, response, choice_submitted.status(), {},
                          cancellation);
      co_return;
    }
    auto choice_events =
        requests_->Events(choice_submitted->request_id, cancellation);
    ChoiceOutcome outcome;
    while (auto next = co_await choice_events.next()) {
      auto event = std::move(*next);
      if (!event.error.ok()) {
        co_await WriteError(request, response, event.error, {}, cancellation);
        co_return;
      }
      outcome.usage = event.usage;
      outcome.text += event.text_delta;
      for (auto& logprob : event.logprobs) {
        outcome.logprobs.push_back(std::move(logprob));
      }
      if (event.terminal) {
        outcome.finish = event.finish_reason;
        break;
      }
    }
    if (outcome.finish == scheduler_client::FinishReason::kNone) {
      co_await WriteError(request, response,
                          InternalError("generation stream ended without a "
                                        "terminal event"),
                          {}, cancellation);
      co_return;
    }
    usage.completion_tokens += outcome.usage.completion_tokens;
    outcomes.push_back(std::move(outcome));
  }

  const ::inferx::api::Usage api_usage{
      static_cast<int32_t>(usage.prompt_tokens),
      static_cast<int32_t>(usage.completion_tokens)};
  const auto finished_at = std::chrono::steady_clock::now();
  const auto first = first_token_at.value_or(finished_at);
  LOG(INFO) << Rid(submitted->request_id)
            << "access route=/v1/chat/completions model=" << parsed->model
            << " status=200 prompt_tokens=" << usage.prompt_tokens
            << " completion_tokens=" << usage.completion_tokens
            << " queue_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   first - queued_at).count()
            << " ttft_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   first - request_started).count()
            << " latency_ms="
            << std::chrono::duration_cast<std::chrono::milliseconds>(
                   finished_at - request_started).count()
            << " finish_reason=" << FinishName(finish);
  if (parsed->sampling.stream) {
    const auto api_finish = ApiFinish(finish);
    (void)co_await sse.Event(
        ::inferx::api::ChatCompletionChunkJson(
            submitted->request_id, parsed->model, {}, {}, &api_finish, created,
            sampled),
        cancellation);
    if (parsed->sampling.include_usage) {
      (void)co_await sse.Event(
          ::inferx::api::UsageChunkJson(submitted->request_id, parsed->model,
                                        api_usage, created, true, sampled),
          cancellation);
    }
    (void)co_await sse.Event("[DONE]", cancellation);
    (void)co_await sse.Finish(cancellation);
    co_return;
  }

  // Reports are decoded into a stable vector first, because the choices only
  // borrow them for the one rendering call.
  std::vector<::inferx::api::ChoiceLogprobs> reports(outcomes.size());
  std::vector<::inferx::api::ChatChoice> choices;
  choices.reserve(outcomes.size());
  for (size_t index = 0; index < outcomes.size(); ++index) {
    ::inferx::api::ChatChoice out_choice;
    out_choice.content = std::move(outcomes[index].text);
    out_choice.finish_reason = ApiFinish(outcomes[index].finish);
    if (parsed->sampling.want_logprobs) {
      auto decoded =
          DecodeLogprobs(tokenization_, version, outcomes[index].logprobs);
      if (!decoded.ok()) {
        co_await WriteError(request, response, decoded.status(), {},
                            cancellation);
        co_return;
      }
      reports[index] = std::move(*decoded);
      out_choice.logprobs = &reports[index];
    }
    choices.push_back(std::move(out_choice));
  }
  transport::HttpResponse result{boost::beast::http::status::ok,
                                 request.version()};
  result.keep_alive(request.keep_alive());
  result.set(boost::beast::http::field::content_type, "application/json");
  result.body() = ::inferx::api::ChatCompletionJson(
      submitted->request_id, parsed->model, choices, api_usage, created,
      sampled);
  result.prepare_payload();
  (void)co_await response.WriteResponse(std::move(result), cancellation);
}

}  // namespace inferx::server::handlers
