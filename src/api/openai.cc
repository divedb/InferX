#include "inferx/api/openai.h"

#include <algorithm>
#include <limits>

#include "inferx/support/json.h"

namespace inferx::api {
namespace {

// OpenAI caps these; we cap them too so that a typo cannot ask the engine for
// four billion tokens and have the scheduler discover it later.
constexpr int32_t kMaxTokensLimit = 32768;
// OpenAI's cap on per-position logprob alternatives, and vLLM's
// --max-logprobs default.
constexpr int32_t kMaxLogprobs = 20;
// Parser-level sanity bound; the handler further bounds n by the server's
// sequence cap, which the parser cannot see.
constexpr int32_t kMaxChoices = 64;

/// Request fields whose feature InferX does not implement. Present and
/// non-null means a clear 400 naming the feature -- the vLLM-compat contract
/// is that nothing is silently ignored. Unknown fields stay ignored, which is
/// OpenAI's own tolerance for forward compatibility.
Status RejectUnsupportedFields(const JsonValue& root) {
  static constexpr struct {
    std::string_view field;
    std::string_view feature;
  } kRejected[] = {
      {"logit_bias", "logit biasing"},
      {"bad_words", "bad-words filtering"},
      {"allowed_token_ids", "token allow-lists"},
      {"prompt_logprobs", "prompt logprobs"},
      {"logprob_token_ids", "explicit logprob token ids"},
      {"truncate_prompt_tokens", "prompt truncation"},
      {"structured_outputs", "structured outputs"},
      {"guided_json", "structured outputs"},
      {"guided_regex", "structured outputs"},
      {"guided_choice", "structured outputs"},
      {"guided_grammar", "structured outputs"},
      {"prompt_embeds", "prompt embeddings"},
  };
  for (const auto& rejected : kRejected) {
    if (const JsonValue* value = root.Find(rejected.field);
        value != nullptr && !value->IsNull()) {
      return InvalidArgumentError(rejected.field,
                                  " is not supported by InferX (",
                                  rejected.feature, ")");
    }
  }

  INFERX_ASSIGN_OR_RETURN(const bool use_beam_search,
                          root.OptionalBool("use_beam_search", false));
  if (use_beam_search) {
    return InvalidArgumentError(
        "use_beam_search is not supported by InferX (beam search)");
  }

  // `{"type": "text"}` asks for what happens anyway; anything else is a
  // structured-output request.
  if (const JsonValue* format = root.Find("response_format");
      format != nullptr && !format->IsNull()) {
    if (!format->IsObject()) {
      return InvalidArgumentError("response_format must be an object, got ",
                                  format->KindName());
    }
    INFERX_ASSIGN_OR_RETURN(const std::string_view type,
                            format->RequiredString("type"));
    if (type != "text") {
      return InvalidArgumentError("response_format type \"", type,
                                  "\" is not supported by InferX "
                                  "(structured outputs)");
    }
  }

  INFERX_ASSIGN_OR_RETURN(
      const bool spaces,
      root.OptionalBool("spaces_between_special_tokens", true));
  if (!spaces) {
    return InvalidArgumentError(
        "spaces_between_special_tokens=false is not supported by InferX");
  }

  return OkStatus();
}

Status ParseSampling(const JsonValue& root, SamplingRequest* out) {
  INFERX_RETURN_IF_ERROR(RejectUnsupportedFields(root));

  INFERX_ASSIGN_OR_RETURN(int64_t max_tokens,
                          root.OptionalInt("max_tokens", out->max_tokens));

  // OpenAI renamed chat's max_tokens to max_completion_tokens; both are
  // accepted, and disagreement is a client bug worth naming rather than a
  // precedence puzzle.
  if (const JsonValue* renamed = root.Find("max_completion_tokens");
      renamed != nullptr && !renamed->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const int64_t value, renamed->AsInt());
    if (root.Find("max_tokens") != nullptr && value != max_tokens) {
      return InvalidArgumentError(
          "max_tokens and max_completion_tokens disagree; send one of them");
    }
    max_tokens = value;
  }

  if (max_tokens <= 0 || max_tokens > kMaxTokensLimit) {
    return InvalidArgumentError("max_tokens must be between 1 and ",
                                kMaxTokensLimit, ", got ", max_tokens);
  }

  out->max_tokens = static_cast<int32_t>(max_tokens);

  INFERX_ASSIGN_OR_RETURN(out->stream, root.OptionalBool("stream", false));

  // `stream_options` is an object, and clients send it whether or not they are
  // streaming. Reject a non-object outright -- a client that sent a bare
  // `true` here is confused about the field and should hear so -- but treat a
  // missing `include_usage` as false rather than an error, since OpenAI has
  // added keys to this object before and will again.
  if (const JsonValue* opts = root.Find("stream_options");
      opts != nullptr && !opts->IsNull()) {
    if (!opts->IsObject()) {
      return InvalidArgumentError("stream_options must be an object, got ",
                                  opts->KindName());
    }

    INFERX_ASSIGN_OR_RETURN(out->include_usage,
                            opts->OptionalBool("include_usage", false));
    if (!out->stream) {
      return InvalidArgumentError(
          "stream_options is valid only when stream is true");
    }
  }

  if (const JsonValue* t = root.Find("temperature");
      t != nullptr && !t->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const double value, t->AsDouble());

    if (value < 0.0 || value > 2.0) {
      return InvalidArgumentError("temperature must be in [0, 2], got ", value);
    }

    out->temperature = static_cast<float>(value);
  }

  if (const JsonValue* p = root.Find("top_p"); p != nullptr && !p->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const double value, p->AsDouble());

    if (value <= 0.0 || value > 1.0) {
      return InvalidArgumentError("top_p must be in (0, 1], got ", value);
    }

    out->top_p = static_cast<float>(value);
  }

  INFERX_ASSIGN_OR_RETURN(const int64_t top_k,
                          root.OptionalInt("top_k", out->top_k));
  // vLLM historically used -1 for "disabled"; treat it as 0.
  if (top_k < -1 || top_k > std::numeric_limits<int32_t>::max()) {
    return InvalidArgumentError("top_k must be a non-negative 32-bit integer");
  }
  out->top_k = top_k < 0 ? 0 : static_cast<int32_t>(top_k);

  if (const JsonValue* min_p = root.Find("min_p");
      min_p != nullptr && !min_p->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const double value, min_p->AsDouble());
    if (value < 0.0 || value > 1.0) {
      return InvalidArgumentError("min_p must be in [0, 1], got ", value);
    }
    out->min_p = static_cast<float>(value);
  }

  if (const JsonValue* penalty = root.Find("presence_penalty");
      penalty != nullptr && !penalty->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const double value, penalty->AsDouble());
    if (value < -2.0 || value > 2.0) {
      return InvalidArgumentError("presence_penalty must be in [-2, 2], got ",
                                  value);
    }
    out->presence_penalty = static_cast<float>(value);
  }

  if (const JsonValue* penalty = root.Find("frequency_penalty");
      penalty != nullptr && !penalty->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const double value, penalty->AsDouble());
    if (value < -2.0 || value > 2.0) {
      return InvalidArgumentError("frequency_penalty must be in [-2, 2], got ",
                                  value);
    }
    out->frequency_penalty = static_cast<float>(value);
  }

  if (const JsonValue* penalty = root.Find("repetition_penalty");
      penalty != nullptr && !penalty->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const double value, penalty->AsDouble());
    if (value <= 0.0 || value > 2.0) {
      return InvalidArgumentError(
          "repetition_penalty must be in (0, 2], got ", value);
    }
    out->repetition_penalty = static_cast<float>(value);
  }

  INFERX_ASSIGN_OR_RETURN(const int64_t n, root.OptionalInt("n", out->n));
  if (n < 1 || n > kMaxChoices) {
    return InvalidArgumentError("n must be between 1 and ", kMaxChoices,
                                ", got ", n);
  }
  out->n = static_cast<int32_t>(n);

  // Old clients: best_of was removed from vLLM but still arrives; it is only
  // meaningful when it differs from n, which InferX does not implement.
  if (const JsonValue* best_of = root.Find("best_of");
      best_of != nullptr && !best_of->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const int64_t value, best_of->AsInt());
    if (value != n) {
      return InvalidArgumentError(
          "best_of differing from n is not supported by InferX");
    }
  }

  if (const JsonValue* seed = root.Find("seed");
      seed != nullptr && !seed->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const int64_t value, seed->AsInt());
    if (value < 0) return InvalidArgumentError("seed must be non-negative");
    out->seed = static_cast<uint64_t>(value);
    out->has_seed = true;
  }

  // `stop` is either a string or an array of them, and both forms are common
  // enough in the wild that accepting only one breaks real clients.
  if (const JsonValue* stop = root.Find("stop");
      stop != nullptr && !stop->IsNull()) {
    if (stop->kind() == JsonValue::Kind::kString) {
      INFERX_ASSIGN_OR_RETURN(const std::string_view value, stop->AsString());
      if (value.empty()) return InvalidArgumentError("stop must not be empty");
      out->stop.emplace_back(value);
    } else if (stop->IsArray()) {
      INFERX_ASSIGN_OR_RETURN(const auto* list, stop->AsArray());

      for (const JsonValue& entry : *list) {
        INFERX_ASSIGN_OR_RETURN(const std::string_view value,
                                entry.AsString());
        if (value.empty()) {
          return InvalidArgumentError("stop entries must not be empty");
        }
        out->stop.emplace_back(value);
      }
    } else {
      return InvalidArgumentError("stop must be a string or an array of "
                                  "strings, got ", stop->KindName());
    }
  }

  if (const JsonValue* ids = root.Find("stop_token_ids");
      ids != nullptr && !ids->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const auto* list, ids->AsArray());
    for (const JsonValue& entry : *list) {
      INFERX_ASSIGN_OR_RETURN(const int64_t value, entry.AsInt());
      if (value < 0 || value > std::numeric_limits<int32_t>::max()) {
        return InvalidArgumentError(
            "stop_token_ids entries must be non-negative 32-bit integers");
      }
      out->stop_token_ids.push_back(static_cast<int32_t>(value));
    }
  }

  INFERX_ASSIGN_OR_RETURN(out->ignore_eos,
                          root.OptionalBool("ignore_eos", false));

  INFERX_ASSIGN_OR_RETURN(const int64_t min_tokens,
                          root.OptionalInt("min_tokens", 0));
  if (min_tokens < 0 || min_tokens > max_tokens) {
    return InvalidArgumentError("min_tokens must be in [0, max_tokens], got ",
                                min_tokens);
  }
  out->min_tokens = static_cast<int32_t>(min_tokens);

  INFERX_ASSIGN_OR_RETURN(out->skip_special_tokens,
                          root.OptionalBool("skip_special_tokens", true));
  INFERX_ASSIGN_OR_RETURN(
      out->include_stop_str_in_output,
      root.OptionalBool("include_stop_str_in_output", false));

  return OkStatus();
}

/// Chat spelling: `logprobs` is a bool, `top_logprobs` the count. OpenAI
/// requires `logprobs: true` before `top_logprobs` means anything.
Status ParseChatLogprobs(const JsonValue& root, SamplingRequest* out) {
  INFERX_ASSIGN_OR_RETURN(out->want_logprobs,
                          root.OptionalBool("logprobs", false));
  INFERX_ASSIGN_OR_RETURN(const int64_t top,
                          root.OptionalInt("top_logprobs", 0));
  if (top < 0 || top > kMaxLogprobs) {
    return InvalidArgumentError("top_logprobs must be in [0, ", kMaxLogprobs,
                                "], got ", top);
  }
  if (top > 0 && !out->want_logprobs) {
    return InvalidArgumentError(
        "top_logprobs requires \"logprobs\": true");
  }
  out->top_logprobs = static_cast<int32_t>(top);
  return OkStatus();
}

/// Completions spelling: `logprobs` is an integer count (or null).
Status ParseCompletionLogprobs(const JsonValue& root, SamplingRequest* out) {
  if (const JsonValue* logprobs = root.Find("logprobs");
      logprobs != nullptr && !logprobs->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const int64_t value, logprobs->AsInt());
    if (value < 0 || value > kMaxLogprobs) {
      return InvalidArgumentError("logprobs must be in [0, ", kMaxLogprobs,
                                  "], got ", value);
    }
    out->want_logprobs = true;
    out->top_logprobs = static_cast<int32_t>(value);
  }
  return OkStatus();
}

void AppendField(std::string_view key, std::string_view value,
                 std::string* out) {
  AppendJsonString(key, out);
  out->push_back(':');
  AppendJsonString(value, out);
}

/// The raw UTF-8 bytes of a token string, as a JSON array of numbers. OpenAI
/// reports these alongside the string because a token need not be a whole
/// character: the string half of such a token is lossy, the bytes are not.
void AppendBytesArray(std::string_view token, std::string* out) {
  out->push_back('[');
  for (size_t index = 0; index < token.size(); ++index) {
    if (index != 0) out->push_back(',');
    *out += std::to_string(static_cast<unsigned char>(token[index]));
  }
  out->push_back(']');
}

/// Chat shape: `{"content":[{"token","logprob","bytes","top_logprobs"}]}`.
/// Emits the full `"logprobs":<value>` member, null when there is no report,
/// so callers keep the pre-logprobs bytes unchanged.
void AppendChatLogprobs(const ChoiceLogprobs* logprobs, std::string* out) {
  *out += "\"logprobs\":";
  if (logprobs == nullptr) {
    *out += "null";
    return;
  }
  *out += "{\"content\":[";
  for (size_t index = 0; index < logprobs->tokens.size(); ++index) {
    const TokenLogprob& token = logprobs->tokens[index];
    if (index != 0) out->push_back(',');
    *out += "{\"token\":";
    AppendJsonString(token.token, out);
    *out += ",\"logprob\":";
    *out += std::to_string(token.logprob);
    *out += ",\"bytes\":";
    AppendBytesArray(token.token, out);
    *out += ",\"top_logprobs\":[";
    for (size_t rank = 0; rank < token.top.size(); ++rank) {
      if (rank != 0) out->push_back(',');
      *out += "{\"token\":";
      AppendJsonString(token.top[rank].first, out);
      *out += ",\"logprob\":";
      *out += std::to_string(token.top[rank].second);
      *out += ",\"bytes\":";
      AppendBytesArray(token.top[rank].first, out);
      out->push_back('}');
    }
    *out += "]}";
  }
  *out += "]}";
}

/// Completions shape: four parallel arrays. `text_offset` counts bytes from
/// the start of the choice's text, so an echoed prompt shifts every entry by
/// the prompt's length -- the offsets locate tokens in what the client
/// actually received.
void AppendCompletionLogprobs(const ChoiceLogprobs* logprobs,
                              size_t text_offset, std::string* out) {
  *out += "\"logprobs\":";
  if (logprobs == nullptr) {
    *out += "null";
    return;
  }
  *out += "{\"tokens\":[";
  for (size_t index = 0; index < logprobs->tokens.size(); ++index) {
    if (index != 0) out->push_back(',');
    AppendJsonString(logprobs->tokens[index].token, out);
  }
  *out += "],\"token_logprobs\":[";
  for (size_t index = 0; index < logprobs->tokens.size(); ++index) {
    if (index != 0) out->push_back(',');
    *out += std::to_string(logprobs->tokens[index].logprob);
  }
  *out += "],\"top_logprobs\":[";
  for (size_t index = 0; index < logprobs->tokens.size(); ++index) {
    const TokenLogprob& token = logprobs->tokens[index];
    if (index != 0) out->push_back(',');
    out->push_back('{');
    for (size_t rank = 0; rank < token.top.size(); ++rank) {
      if (rank != 0) out->push_back(',');
      AppendJsonString(token.top[rank].first, out);
      out->push_back(':');
      *out += std::to_string(token.top[rank].second);
    }
    out->push_back('}');
  }
  *out += "],\"text_offset\":[";
  size_t offset = text_offset;
  for (size_t index = 0; index < logprobs->tokens.size(); ++index) {
    if (index != 0) out->push_back(',');
    *out += std::to_string(offset);
    offset += logprobs->tokens[index].token.size();
  }
  *out += "]}";
}

}  // namespace

const char* FinishReasonName(FinishReason reason) {
  switch (reason) {
    case FinishReason::kStop:
      return "stop";
    case FinishReason::kLength:
      return "length";
  }

  return "stop";
}

StatusOr<ChatCompletionRequest> ParseChatCompletionRequest(
    std::string_view body) {
  INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(body));

  if (!root.IsObject()) {
    return InvalidArgumentError("request body must be a JSON object, got ",
                                root.KindName());
  }

  ChatCompletionRequest request;

  INFERX_ASSIGN_OR_RETURN(const std::string_view model,
                          root.RequiredString("model"));
  if (model.empty()) return InvalidArgumentError("\"model\" is empty");
  request.model = std::string(model);

  const JsonValue* messages = root.Find("messages");

  if (messages == nullptr) {
    return InvalidArgumentError("missing required field \"messages\"");
  }

  INFERX_ASSIGN_OR_RETURN(const auto* list, messages->AsArray());

  if (list->empty()) return InvalidArgumentError("\"messages\" is empty");

  for (const JsonValue& entry : *list) {
    tokenizer::ChatMessage message;

    INFERX_ASSIGN_OR_RETURN(const std::string_view role,
                            entry.RequiredString("role"));
    message.role = std::string(role);

    // A null content is what a tool-call turn looks like, and we do not render
    // those; anything else non-string is simply malformed.
    INFERX_ASSIGN_OR_RETURN(const std::string_view content,
                            entry.RequiredString("content"));
    message.content = std::string(content);

    request.messages.push_back(std::move(message));
  }

  // Tool calling has server-side plumbing InferX lacks; a request that sends
  // tools expects tool_calls back and must hear "no" rather than get prose.
  if (const JsonValue* tools = root.Find("tools");
      tools != nullptr && !tools->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const auto* list, tools->AsArray());
    if (!list->empty()) {
      return InvalidArgumentError(
          "tools are not supported by InferX (tool calling)");
    }
  }
  if (const JsonValue* choice = root.Find("tool_choice");
      choice != nullptr && !choice->IsNull()) {
    if (choice->kind() != JsonValue::Kind::kString) {
      return InvalidArgumentError(
          "tool_choice is not supported by InferX (tool calling)");
    }
    INFERX_ASSIGN_OR_RETURN(const std::string_view value, choice->AsString());
    if (value != "none") {
      return InvalidArgumentError(
          "tool_choice \"", value,
          "\" is not supported by InferX (tool calling)");
    }
  }

  INFERX_ASSIGN_OR_RETURN(const bool echo, root.OptionalBool("echo", false));
  if (echo) {
    return InvalidArgumentError(
        "echo is not supported on chat completions; use /v1/completions");
  }

  INFERX_RETURN_IF_ERROR(ParseSampling(root, &request.sampling));
  INFERX_RETURN_IF_ERROR(ParseChatLogprobs(root, &request.sampling));

  return request;
}

StatusOr<CompletionRequest> ParseCompletionRequest(std::string_view body) {
  INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(body));

  if (!root.IsObject()) {
    return InvalidArgumentError("request body must be a JSON object, got ",
                                root.KindName());
  }

  CompletionRequest request;

  INFERX_ASSIGN_OR_RETURN(const std::string_view model,
                          root.RequiredString("model"));
  if (model.empty()) return InvalidArgumentError("\"model\" is empty");
  request.model = std::string(model);

  const JsonValue* prompt = root.Find("prompt");

  if (prompt == nullptr) {
    return InvalidArgumentError("missing required field \"prompt\"");
  }

  // The array-of-prompts form is a batch request, which the engine would have
  // to fan out and reassemble. Rejecting it is better than serving only the
  // first element and returning a response that looks complete.
  if (prompt->IsArray()) {
    return UnimplementedError(
        "\"prompt\" as an array is a batch request, which this server does not "
        "implement; send one prompt per request");
  }

  INFERX_ASSIGN_OR_RETURN(const std::string_view text, prompt->AsString());
  request.prompt = std::string(text);

  if (const JsonValue* suffix = root.Find("suffix");
      suffix != nullptr && !suffix->IsNull()) {
    return InvalidArgumentError(
        "suffix is not supported by InferX (fill-in-the-middle)");
  }

  INFERX_RETURN_IF_ERROR(ParseSampling(root, &request.sampling));
  INFERX_RETURN_IF_ERROR(ParseCompletionLogprobs(root, &request.sampling));
  INFERX_ASSIGN_OR_RETURN(request.sampling.echo,
                          root.OptionalBool("echo", false));

  return request;
}

StatusOr<TokenizeRequest> ParseTokenizeRequest(std::string_view body) {
  INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(body));
  if (!root.IsObject()) {
    return InvalidArgumentError("request body must be a JSON object, got ",
                                root.KindName());
  }
  TokenizeRequest request;
  INFERX_ASSIGN_OR_RETURN(const std::string_view model,
                          root.RequiredString("model"));
  if (model.empty()) return InvalidArgumentError("\"model\" is empty");
  request.model = std::string(model);
  INFERX_ASSIGN_OR_RETURN(const std::string_view text,
                          root.RequiredString("text"));
  request.text = std::string(text);
  INFERX_ASSIGN_OR_RETURN(
      request.add_special_tokens,
      root.OptionalBool("add_special_tokens", false));
  return request;
}

StatusOr<VllmTokenizeRequest> ParseVllmTokenizeRequest(std::string_view body) {
  INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(body));
  if (!root.IsObject()) {
    return InvalidArgumentError("request body must be a JSON object, got ",
                                root.KindName());
  }
  VllmTokenizeRequest request;
  INFERX_ASSIGN_OR_RETURN(const std::string_view model,
                          root.RequiredString("model"));
  if (model.empty()) return InvalidArgumentError("\"model\" is empty");
  request.model = std::string(model);
  // vLLM's /tokenize also accepts a chat `messages` form, which requires a
  // chat template; rejecting it loudly beats tokenizing the wrong thing.
  if (const JsonValue* messages = root.Find("messages");
      messages != nullptr && !messages->IsNull()) {
    return InvalidArgumentError(
        "chat tokenization via /tokenize is not supported by InferX; "
        "use \"prompt\"");
  }
  INFERX_ASSIGN_OR_RETURN(const std::string_view prompt,
                          root.RequiredString("prompt"));
  request.prompt = std::string(prompt);
  INFERX_ASSIGN_OR_RETURN(
      request.add_special_tokens,
      root.OptionalBool("add_special_tokens", true));
  return request;
}

StatusOr<DetokenizeRequest> ParseDetokenizeRequest(std::string_view body) {
  INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(body));
  if (!root.IsObject()) {
    return InvalidArgumentError("request body must be a JSON object, got ",
                                root.KindName());
  }
  DetokenizeRequest request;
  INFERX_ASSIGN_OR_RETURN(const std::string_view model,
                          root.RequiredString("model"));
  if (model.empty()) return InvalidArgumentError("\"model\" is empty");
  request.model = std::string(model);
  const JsonValue* tokens = root.Find("tokens");
  if (tokens == nullptr) {
    return InvalidArgumentError("missing required field \"tokens\"");
  }
  if (!tokens->IsArray()) {
    return InvalidArgumentError("\"tokens\" must be an array of integers");
  }
  INFERX_ASSIGN_OR_RETURN(const auto* values, tokens->AsArray());
  request.tokens.reserve(values->size());
  for (const JsonValue& value : *values) {
    INFERX_ASSIGN_OR_RETURN(const int64_t id, value.AsInt());
    if (id < std::numeric_limits<int32_t>::min() ||
        id > std::numeric_limits<int32_t>::max()) {
      return InvalidArgumentError("\"tokens\" entries must be 32-bit integers");
    }
    request.tokens.push_back(static_cast<int32_t>(id));
  }
  return request;
}

StatusOr<EmbeddingsRequest> ParseEmbeddingsRequest(std::string_view body) {
  INFERX_ASSIGN_OR_RETURN(const JsonValue root, ParseJson(body));
  if (!root.IsObject()) {
    return InvalidArgumentError("request body must be a JSON object, got ",
                                root.KindName());
  }
  EmbeddingsRequest request;
  INFERX_ASSIGN_OR_RETURN(const std::string_view model,
                          root.RequiredString("model"));
  if (model.empty()) return InvalidArgumentError("\"model\" is empty");
  request.model = std::string(model);

  const JsonValue* input = root.Find("input");
  if (input == nullptr) return InvalidArgumentError("missing required field \"input\"");
  if (input->kind() == JsonValue::Kind::kString) {
    INFERX_ASSIGN_OR_RETURN(const std::string_view text, input->AsString());
    request.input.emplace_back(text);
  } else if (input->IsArray()) {
    INFERX_ASSIGN_OR_RETURN(const auto* values, input->AsArray());
    if (values->empty()) return InvalidArgumentError("\"input\" is empty");
    for (const JsonValue& value : *values) {
      INFERX_ASSIGN_OR_RETURN(const std::string_view text, value.AsString());
      request.input.emplace_back(text);
    }
  } else {
    return InvalidArgumentError("input must be a string or array of strings");
  }

  if (const JsonValue* encoding = root.Find("encoding_format");
      encoding != nullptr && !encoding->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const std::string_view value, encoding->AsString());
    if (value != "float" && value != "base64") {
      return InvalidArgumentError("encoding_format must be float or base64");
    }
    request.encoding_format = std::string(value);
  }
  if (const JsonValue* dimensions = root.Find("dimensions");
      dimensions != nullptr && !dimensions->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const int64_t value, dimensions->AsInt());
    if (value <= 0 || value > std::numeric_limits<int32_t>::max()) {
      return InvalidArgumentError("dimensions must be a positive 32-bit integer");
    }
    request.dimensions = static_cast<int32_t>(value);
    request.has_dimensions = true;
  }
  return request;
}

std::string ChatCompletionJson(std::string_view id, std::string_view model,
                               const std::vector<ChatChoice>& choices,
                               const Usage& usage, int64_t created,
                               bool sampled) {
  std::string out = "{";

  AppendField("id", id, &out);
  out += ",\"object\":\"chat.completion\",\"created\":";
  out += std::to_string(created);
  out += ',';
  AppendField("model", model, &out);
  out += ",\"system_fingerprint\":";
  AppendJsonString(sampled ? "sampled" : "greedy", &out);
  out += ",\"choices\":[";
  for (size_t index = 0; index < choices.size(); ++index) {
    if (index != 0) out.push_back(',');
    out += "{\"index\":";
    out += std::to_string(index);
    out += ",\"message\":{\"role\":\"assistant\",\"content\":";
    AppendJsonString(choices[index].content, &out);
    out += "},";
    AppendChatLogprobs(choices[index].logprobs, &out);
    out += ",\"finish_reason\":";
    AppendJsonString(FinishReasonName(choices[index].finish_reason), &out);
    out.push_back('}');
  }
  out += "],\"usage\":{\"prompt_tokens\":";
  out += std::to_string(usage.prompt_tokens);
  out += ",\"completion_tokens\":";
  out += std::to_string(usage.completion_tokens);
  out += ",\"total_tokens\":";
  out += std::to_string(usage.prompt_tokens + usage.completion_tokens);
  out += "}}";

  return out;
}

std::string ChatCompletionChunkJson(std::string_view id, std::string_view model,
                                    std::string_view role,
                                    std::string_view content,
                                    const FinishReason* reason,
                                    int64_t created, bool sampled,
                                    const ChoiceLogprobs* logprobs) {
  std::string out = "{";

  AppendField("id", id, &out);
  out += ",\"object\":\"chat.completion.chunk\",\"created\":";
  out += std::to_string(created);
  out += ',';
  AppendField("model", model, &out);
  out += ",\"system_fingerprint\":";
  AppendJsonString(sampled ? "sampled" : "greedy", &out);
  out += ",\"choices\":[{\"index\":0,\"delta\":{";

  bool first = true;

  if (!role.empty()) {
    AppendField("role", role, &out);
    first = false;
  }

  // The final chunk carries an empty delta and the finish reason; emitting a
  // "content" key there would make some clients append an empty string, which
  // is harmless, and others treat it as a token, which is not.
  if (reason == nullptr) {
    if (!first) out.push_back(',');
    AppendField("content", content, &out);
  }

  out += "},";
  AppendChatLogprobs(logprobs, &out);
  out += ",\"finish_reason\":";

  if (reason == nullptr) {
    out += "null";
  } else {
    AppendJsonString(FinishReasonName(*reason), &out);
  }

  out += "}]}";

  return out;
}

std::string CompletionJson(std::string_view id, std::string_view model,
                           const std::vector<CompletionChoice>& choices,
                           const Usage& usage, int64_t created, bool sampled) {
  std::string out = "{";

  AppendField("id", id, &out);
  out += ",\"object\":\"text_completion\",\"created\":";
  out += std::to_string(created);
  out += ',';
  AppendField("model", model, &out);
  out += ",\"system_fingerprint\":";
  AppendJsonString(sampled ? "sampled" : "greedy", &out);
  out += ",\"choices\":[";
  for (size_t index = 0; index < choices.size(); ++index) {
    if (index != 0) out.push_back(',');
    out += "{\"index\":";
    out += std::to_string(index);
    out += ",\"text\":";
    AppendJsonString(choices[index].text, &out);
    out.push_back(',');
    AppendCompletionLogprobs(choices[index].logprobs,
                             choices[index].text_offset, &out);
    out += ",\"finish_reason\":";
    AppendJsonString(FinishReasonName(choices[index].finish_reason), &out);
    out.push_back('}');
  }
  out += "],\"usage\":{\"prompt_tokens\":";
  out += std::to_string(usage.prompt_tokens);
  out += ",\"completion_tokens\":";
  out += std::to_string(usage.completion_tokens);
  out += ",\"total_tokens\":";
  out += std::to_string(usage.prompt_tokens + usage.completion_tokens);
  out += "}}";

  return out;
}

std::string CompletionChunkJson(std::string_view id, std::string_view model,
                                std::string_view text,
                                const FinishReason* reason, int64_t created,
                                bool sampled, const ChoiceLogprobs* logprobs,
                                size_t text_offset) {
  std::string out = "{";

  AppendField("id", id, &out);
  out += ",\"object\":\"text_completion\",\"created\":";
  out += std::to_string(created);
  out += ',';
  AppendField("model", model, &out);
  out += ",\"system_fingerprint\":";
  AppendJsonString(sampled ? "sampled" : "greedy", &out);
  out += ",\"choices\":[{\"index\":0,\"text\":";
  AppendJsonString(text, &out);
  out.push_back(',');
  AppendCompletionLogprobs(logprobs, text_offset, &out);
  out += ",\"finish_reason\":";

  if (reason == nullptr) {
    out += "null";
  } else {
    AppendJsonString(FinishReasonName(*reason), &out);
  }

  out += "}]}";

  return out;
}

std::string UsageChunkJson(std::string_view id, std::string_view model,
                           const Usage& usage, int64_t created, bool chat,
                           bool sampled) {
  std::string out = "{";

  AppendField("id", id, &out);
  out += chat ? ",\"object\":\"chat.completion.chunk\",\"created\":"
              : ",\"object\":\"text_completion\",\"created\":";
  out += std::to_string(created);
  out += ',';
  AppendField("model", model, &out);
  out += ",\"system_fingerprint\":";
  AppendJsonString(sampled ? "sampled" : "greedy", &out);
  out += ",\"choices\":[],\"usage\":{\"prompt_tokens\":";
  out += std::to_string(usage.prompt_tokens);
  out += ",\"completion_tokens\":";
  out += std::to_string(usage.completion_tokens);
  out += ",\"total_tokens\":";
  out += std::to_string(usage.prompt_tokens + usage.completion_tokens);
  out += "}}";

  return out;
}

std::string ModelsJson(std::string_view model, int64_t created) {
  std::string out = "{\"object\":\"list\",\"data\":[{";

  AppendField("id", model, &out);
  out += ",\"object\":\"model\",\"created\":";
  out += std::to_string(created);
  out += ",\"owned_by\":\"inferx\"}]}";

  return out;
}

std::string TokenizeJson(std::string_view model,
                         const std::vector<int32_t>& token_ids) {
  std::string out = "{\"model\":";
  AppendJsonString(model, &out);
  out += ",\"token_ids\":[";
  for (size_t index = 0; index < token_ids.size(); ++index) {
    if (index != 0) out.push_back(',');
    out += std::to_string(token_ids[index]);
  }
  out += "],\"token_count\":";
  out += std::to_string(token_ids.size());
  out += "}";
  return out;
}

std::string VllmTokenizeJson(const std::vector<int32_t>& token_ids) {
  std::string out = "{\"count\":";
  out += std::to_string(token_ids.size());
  out += ",\"max_model_len\":null,\"tokens\":[";
  for (size_t index = 0; index < token_ids.size(); ++index) {
    if (index != 0) out.push_back(',');
    out += std::to_string(token_ids[index]);
  }
  out += "]}";
  return out;
}

std::string DetokenizeJson(std::string_view prompt) {
  std::string out = "{\"prompt\":";
  AppendJsonString(prompt, &out);
  out += "}";
  return out;
}

std::string ErrorJson(std::string_view message, std::string_view type) {
  std::string out = "{\"error\":{";

  AppendField("message", message, &out);
  out.push_back(',');
  AppendField("type", type, &out);
  out += ",\"param\":null,\"code\":null}}";

  return out;
}

std::string SseFrame(std::string_view data) {
  std::string out;
  out.reserve(data.size() + 8);

  // Each line of the payload becomes its own `data:` line, per the SSE grammar.
  size_t start = 0;

  while (start <= data.size()) {
    const size_t end = data.find('\n', start);
    const size_t stop = end == std::string_view::npos ? data.size() : end;

    out += "data: ";
    out.append(data.substr(start, stop - start));
    out.push_back('\n');

    if (end == std::string_view::npos) break;

    start = end + 1;
  }

  out.push_back('\n');

  return out;
}

size_t FindStopSequence(std::string_view text,
                        const std::vector<std::string>& stop) {
  size_t best = std::string_view::npos;

  for (const std::string& needle : stop) {
    if (needle.empty()) continue;

    if (const size_t at = text.find(needle); at != std::string_view::npos) {
      best = std::min(best, at);
    }
  }

  return best;
}

size_t StopSequenceHoldback(std::string_view text,
                            const std::vector<std::string>& stop) {
  size_t hold = 0;

  for (const std::string& needle : stop) {
    if (needle.empty()) continue;

    // The longest proper prefix of `needle` that is also a suffix of `text`.
    // Anything shorter than the whole needle is still undecided, so it has to
    // be withheld; the whole needle is handled by FindStopSequence instead.
    const size_t limit = std::min(text.size(), needle.size() - 1);

    for (size_t length = limit; length > 0; --length) {
      if (text.compare(text.size() - length, length, needle, 0, length) == 0) {
        hold = std::max(hold, length);
        break;
      }
    }
  }

  return hold;
}

}  // namespace inferx::api
