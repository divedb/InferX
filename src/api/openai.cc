#include "inferx/api/openai.h"

#include <algorithm>
#include <limits>

#include "inferx/support/json.h"

namespace inferx::api {
namespace {

// OpenAI caps these; we cap them too so that a typo cannot ask the engine for
// four billion tokens and have the scheduler discover it later.
constexpr int32_t kMaxTokensLimit = 32768;

Status ParseSampling(const JsonValue& root, SamplingRequest* out) {
  INFERX_ASSIGN_OR_RETURN(const int64_t max_tokens,
                          root.OptionalInt("max_tokens", out->max_tokens));

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
  if (top_k < 0 || top_k > std::numeric_limits<int32_t>::max()) {
    return InvalidArgumentError("top_k must be a non-negative 32-bit integer");
  }
  out->top_k = static_cast<int32_t>(top_k);
  if (out->top_k != 0) {
    return UnimplementedError("top_k is not supported by the current sampler");
  }

  if (const JsonValue* penalty = root.Find("repetition_penalty");
      penalty != nullptr && !penalty->IsNull()) {
    INFERX_ASSIGN_OR_RETURN(const double value, penalty->AsDouble());
    if (value <= 0.0 || value > 2.0) {
      return InvalidArgumentError(
          "repetition_penalty must be in (0, 2], got ", value);
    }
    out->repetition_penalty = static_cast<float>(value);
    if (out->repetition_penalty != 1.0f) {
      return UnimplementedError(
          "repetition_penalty is not supported by the current sampler");
    }
  }

  INFERX_ASSIGN_OR_RETURN(const int64_t n, root.OptionalInt("n", out->n));
  if (n != 1) {
    return UnimplementedError("n must be 1; multiple sequences are unsupported");
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

  return OkStatus();
}

void AppendField(std::string_view key, std::string_view value,
                 std::string* out) {
  AppendJsonString(key, out);
  out->push_back(':');
  AppendJsonString(value, out);
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

  INFERX_RETURN_IF_ERROR(ParseSampling(root, &request.sampling));

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

  INFERX_RETURN_IF_ERROR(ParseSampling(root, &request.sampling));

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
                               std::string_view content, FinishReason reason,
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
  out += ",\"choices\":[{\"index\":0,"
         "\"message\":{\"role\":\"assistant\",\"content\":";
  AppendJsonString(content, &out);
  out += "},\"logprobs\":null,\"finish_reason\":";
  AppendJsonString(FinishReasonName(reason), &out);
  out += "}],\"usage\":{\"prompt_tokens\":";
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
                                    int64_t created, bool sampled) {
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

  out += "},\"logprobs\":null,\"finish_reason\":";

  if (reason == nullptr) {
    out += "null";
  } else {
    AppendJsonString(FinishReasonName(*reason), &out);
  }

  out += "}]}";

  return out;
}

std::string CompletionJson(std::string_view id, std::string_view model,
                           std::string_view text, FinishReason reason,
                           const Usage& usage, int64_t created, bool sampled) {
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
  out += ",\"logprobs\":null,\"finish_reason\":";
  AppendJsonString(FinishReasonName(reason), &out);
  out += "}],\"usage\":{\"prompt_tokens\":";
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
                                bool sampled) {
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
  out += ",\"logprobs\":null,\"finish_reason\":";

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
