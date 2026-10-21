// Implementation of the OpenAI-compatible API schemas (see api.h) and the
// stable ApiError model (see api_error.h).
#include "inferx/server/api.h"

#include <cmath>
#include <limits>
#include <random>
#include <stdexcept>

#include "inferx/server/api_error.h"

namespace inferx {
namespace {

ParseResult Failure(int status, std::string message, std::string param = {},
                    std::string code = {}) {
  ParseResult out;
  out.http_status = status;
  out.message = std::move(message);
  out.param = std::move(param);
  out.code = std::move(code);
  return out;
}

bool FiniteNumber(const Json& j) { return j.is_number() && std::isfinite(j.get<double>()); }

/// JSON string serialization that replaces invalid UTF-8 instead of
/// throwing: token streams can transiently end mid-multibyte character.
std::string SafeDump(std::string_view text) {
  return Json(std::string(text)).dump(-1, ' ', false,
                                       Json::error_handler_t::replace);
}

/// Reads a finite number field into `out` when present; returns a 400 with
/// `param` on a non-numeric or non-finite value.
template <class T>
bool ReadNumber(const Json& j, const char* field, T* out, ParseResult* err) {
  if (!j.contains(field)) return true;
  const Json& v = j[field];
  if (!FiniteNumber(v)) {
    *err = Failure(400, std::string("field '") + field + "' must be a finite number", field);
    return false;
  }
  *out = static_cast<T>(v.get<double>());
  return true;
}

}  // namespace

ParseResult ParseCompletionsRequest(std::string_view body,
                                    const sampling::SamplingParams& defaults,
                                    const std::string& expected_model) {
  Json j;
  try {
    j = Json::parse(body);
  } catch (const std::exception& e) {
    return Failure(400, std::string("invalid JSON body: ") + e.what(), "", "invalid_json");
  }
  if (!j.is_object()) {
    return Failure(400, "request body must be a JSON object", "", "invalid_json");
  }

  if (j.contains("model") && j["model"].is_string()) {
    const std::string model = j["model"].get<std::string>();
    if (model != expected_model) {
      return Failure(404, "model '" + model + "' not found", "model", "model_not_found");
    }
  }

  ParseResult out;
  if (!j.contains("prompt")) {
    return Failure(400, "field 'prompt' is required", "prompt");
  }
  const Json& prompt = j["prompt"];
  if (prompt.is_string()) {
    out.request.prompt = prompt.get<std::string>();
  } else if (prompt.is_array()) {
    // vLLM accepts pre-tokenized prompts as an integer array; arrays of
    // strings (multiple prompts) are not supported yet.
    std::vector<TokenId> ids;
    ids.reserve(prompt.size());
    for (const Json& v : prompt) {
      if (!v.is_number_integer() || v.get<long long>() < 0) {
        return Failure(400,
                       "field 'prompt' arrays must contain non-negative token ids",
                       "prompt", "unsupported_parameter");
      }
      ids.push_back(static_cast<TokenId>(v.get<long long>()));
    }
    out.request.token_ids = std::move(ids);
  } else {
    return Failure(400, "field 'prompt' must be a string or a token id array",
                   "prompt");
  }

  // Everything else merges over the server's defaults.
  sampling::SamplingParams p = defaults;
  if (j.contains("stop") && !j["stop"].is_null()) {
    const auto& stop = j["stop"];
    if (!stop.is_string() && !stop.is_array()) {
      return Failure(400, "stop must be a string or an array of strings", "stop");
    }
    const Json stops = stop.is_string() ? Json::array({stop}) : stop;
    if (stops.size() > 4) return Failure(400, "at most four stop strings are supported", "stop");
    p.stop_strings.clear();
    for (const auto& value : stops) {
      if (!value.is_string() || value.get_ref<const std::string&>().empty()) {
        return Failure(400, "stop strings must be non-empty strings", "stop");
      }
      p.stop_strings.push_back(value.get<std::string>());
    }
  }
  if (j.contains("stop_token_ids") && !j["stop_token_ids"].is_null()) {
    const auto& ids = j["stop_token_ids"];
    if (!ids.is_array()) return Failure(400, "stop_token_ids must be an array", "stop_token_ids");
    p.stop_token_ids.clear();
    for (const auto& id : ids) {
      if (!id.is_number_integer() || id.get<double>() < 0 ||
          id.get<double>() > std::numeric_limits<TokenId>::max()) {
        return Failure(400, "stop_token_ids must contain non-negative int32 token ids",
                       "stop_token_ids");
      }
      p.stop_token_ids.push_back(id.get<TokenId>());
    }
  }
  ParseResult err;
  double number = 0;
  if (j.contains("max_tokens")) {
    const Json& v = j["max_tokens"];
    if (!v.is_number_unsigned() || v.get<long long>() < 1) {
      return Failure(400, "field 'max_tokens' must be an integer >= 1", "max_tokens");
    }
    p.max_tokens = static_cast<std::uint32_t>(v.get<long long>());
  }
  if (!ReadNumber(j, "temperature", &number, &err)) return err;
  if (j.contains("temperature")) p.temperature = static_cast<float>(number);
  if (!ReadNumber(j, "top_p", &number, &err)) return err;
  if (j.contains("top_p")) p.top_p = static_cast<float>(number);
  if (!ReadNumber(j, "min_p", &number, &err)) return err;
  if (j.contains("min_p")) p.min_p = static_cast<float>(number);
  if (!ReadNumber(j, "repetition_penalty", &number, &err)) return err;
  if (j.contains("repetition_penalty")) p.repetition_penalty = static_cast<float>(number);
  if (!ReadNumber(j, "frequency_penalty", &number, &err)) return err;
  if (j.contains("frequency_penalty")) p.frequency_penalty = static_cast<float>(number);
  if (!ReadNumber(j, "presence_penalty", &number, &err)) return err;
  if (j.contains("presence_penalty")) p.presence_penalty = static_cast<float>(number);
  if (j.contains("top_k")) {
    const Json& v = j["top_k"];
    if (!v.is_number() || v.get<double>() < 0) {
      return Failure(400, "field 'top_k' must be a non-negative number", "top_k");
    }
    p.top_k = static_cast<std::uint32_t>(v.get<double>());
  }
  if (j.contains("seed")) {
    const Json& v = j["seed"];
    if (!v.is_number_integer()) {
      return Failure(400, "field 'seed' must be an integer", "seed");
    }
    p.seed = v.get<std::uint64_t>();
  }
  if (j.contains("ignore_eos")) {
    if (!j["ignore_eos"].is_boolean()) {
      return Failure(400, "field 'ignore_eos' must be a boolean", "ignore_eos");
    }
    p.ignore_eos = j["ignore_eos"].get<bool>();
  }
  if (j.contains("stream")) {
    if (!j["stream"].is_boolean()) {
      return Failure(400, "field 'stream' must be a boolean", "stream");
    }
    out.request.stream = j["stream"].get<bool>();
  }
  // Accepted and ignored: usage is always reported on the final event;
  // include_usage only controls the extra usage-only SSE chunk (vLLM
  // behavior).
  if (j.contains("stream_options") && !j["stream_options"].is_object()) {
    return Failure(400, "field 'stream_options' must be an object", "stream_options");
  }
  if (j.contains("stream_options") && j["stream_options"].contains("include_usage")) {
    const Json& v = j["stream_options"]["include_usage"];
    if (!v.is_boolean()) {
      return Failure(400, "field 'stream_options.include_usage' must be a boolean",
                     "stream_options");
    }
    out.request.include_usage = v.get<bool>();
  }
  // Known generation controls must never silently change meaning. Keep
  // unknown extension fields forward-compatible, but reject these until
  // their behavior is implemented end to end (not just in SamplingParams).
  for (const char* unsupported : {
           "echo", "suffix", "best_of",
           "include_stop_str_in_output", "min_tokens", "logit_bias",
           "allowed_token_ids", "prompt_logprobs", "top_logprobs",
           "response_format", "structured_outputs", "guided_json",
           "guided_regex", "guided_choice", "guided_grammar",
           "guided_whitespace_pattern", "guided_decoding_backend",
           "tools", "tool_choice", "parallel_tool_calls",
           "max_completion_tokens", "truncate_prompt_tokens",
           "return_token_ids", "return_tokens_as_token_ids",
           "use_beam_search", "length_penalty", "early_stopping"}) {
    if (j.contains(unsupported)) {
      return Failure(400, std::string("field '") + unsupported + "' is not supported",
                     unsupported, "unsupported_parameter");
    }
  }
  if (j.contains("logprobs") && !j["logprobs"].is_null()) {
    return Failure(400, "logprobs are not supported yet", "logprobs", "not_supported");
  }
  if (j.contains("n") && (!j["n"].is_number_unsigned() || j["n"].get<long long>() != 1)) {
    return Failure(400, "only n = 1 is supported", "n", "not_supported");
  }

  const Status validated = p.Validate();
  if (!validated.ok()) {
    return Failure(400, std::string(validated.message()), "sampling");
  }
  out.request.params = std::move(p);
  out.ok = true;
  return out;
}

ParseResult ParseChatRequest(std::string_view body,
                             const sampling::SamplingParams& defaults,
                             const std::string& expected_model) {
  Json j;
  try { j = Json::parse(body); }
  catch (const std::exception&) { return Failure(400, "invalid JSON body", "", "invalid_json"); }
  if (!j.is_object()) return Failure(400, "request body must be an object", "", "invalid_json");
  if (j.contains("model") && !j["model"].is_string())
    return Failure(400, "model must be a string", "model");
  if (j.contains("model") && j["model"] != expected_model)
    return Failure(404, "model not found", "model", "model_not_found");
  if (!j.contains("messages") || !j["messages"].is_array() || j["messages"].empty())
    return Failure(400, "messages must be a non-empty array", "messages");
  ChatPrompt chat;
  for (const auto& m : j["messages"]) {
    if (!m.is_object() || !m.contains("role") || !m["role"].is_string() ||
        !m.contains("content")) return Failure(400, "invalid chat message", "messages");
    for (auto it = m.begin(); it != m.end(); ++it) {
      if (it.key() != "role" && it.key() != "content")
        return Failure(400, "unsupported message field: " + it.key(), "messages." + it.key(), "unsupported_parameter");
    }
    ChatMessage message;
    const auto role = m["role"].get<std::string>();
    if (role == "system") message.role = Role::kSystem;
    else if (role == "user") message.role = Role::kUser;
    else if (role == "assistant") message.role = Role::kAssistant;
    else return Failure(400, "unsupported message role", "messages.role", "unsupported_parameter");
    if (!m["content"].is_string())
      return Failure(400, "only string message content is supported", "messages.content", "unsupported_parameter");
    message.content.push_back(TextPart{m["content"].get<std::string>()});
    chat.messages.push_back(std::move(message));
  }
  if (j.contains("chat_template_kwargs")) {
    const auto& kwargs = j["chat_template_kwargs"];
    if (!kwargs.is_object()) return Failure(400, "chat_template_kwargs must be an object", "chat_template_kwargs");
    for (auto it = kwargs.begin(); it != kwargs.end(); ++it) {
      if (it.key() != "enable_thinking")
        return Failure(400, "unsupported chat template option", "chat_template_kwargs." + it.key(), "unsupported_parameter");
      if (!it.value().is_boolean()) return Failure(400, "enable_thinking must be boolean", "chat_template_kwargs.enable_thinking");
      chat.options.enable_thinking = it.value().get<bool>();
    }
  }
  for (const char* field : {"prompt", "chat_template", "add_generation_prompt", "continue_final_message", "reasoning_effort", "function_call", "functions"}) {
    if (j.contains(field)) return Failure(400, std::string("unsupported chat field: ") + field, field, "unsupported_parameter");
  }
  if (j.contains("max_completion_tokens")) {
    if (j.contains("max_tokens")) return Failure(400, "specify only one token limit", "max_completion_tokens");
    j["max_tokens"] = j["max_completion_tokens"];
    j.erase("max_completion_tokens");
  }
  if (j.contains("logprobs") && j["logprobs"] == false) j["logprobs"] = nullptr;
  // Reuse exactly the same sampling and unsupported-control validation.
  j.erase("messages");
  j.erase("chat_template_kwargs");
  j["prompt"] = "";
  auto parsed = ParseCompletionsRequest(j.dump(), defaults, expected_model);
  if (parsed.ok) parsed.request.chat = std::move(chat);
  return parsed;
}

std::string_view FinishReasonName(FinishReason reason) {
  switch (reason) {
    case FinishReason::kLengthCapped:
      return "length";
    case FinishReason::kStopped:
    default:
      return "stop";
  }
}

std::string NewRequestId(std::string_view prefix) {
  static std::mt19937_64 rng{std::random_device{}()};
  static const char* kHex = "0123456789abcdef";
  std::string id(prefix);
  id.reserve(prefix.size() + 24);
  for (int i = 0; i < 24; ++i) {
    id += kHex[rng() & 0xF];
  }
  return id;
}

Json MakeError(int status, std::string message, std::string param, std::string code) {
  // OpenAI error types keyed by status (vLLM emits the same envelope).
  const char* type = "invalid_request_error";
  if (status == 401) type = "authentication_error";
  if (status == 429 || status == 503) type = "server_error";
  if (status >= 500) type = "server_error";
  if (status == 501) type = "not_implemented_error";
  Json error = Json::object();
  error["message"] = std::move(message);
  error["type"] = type;
  error["param"] = param.empty() ? Json() : Json(std::move(param));
  error["code"] = code.empty() ? Json() : Json(std::move(code));
  Json body = Json::object();
  body["error"] = std::move(error);
  return body;
}

Json MakeModelList(const std::string& model, std::int64_t created) {
  Json entry = Json::object();
  entry["id"] = model;
  entry["object"] = "model";
  entry["created"] = created;
  entry["owned_by"] = "inferx";
  Json body = Json::object();
  body["object"] = "list";
  body["data"] = Json::array({std::move(entry)});
  return body;
}

Json MakeCompletionResponse(const std::string& id, const std::string& model,
                            std::int64_t created, const std::string& text,
                            std::string_view finish_reason, int prompt_tokens,
                            int completion_tokens, bool chat) {
  Json choice = Json::object();
  if (chat) choice["message"] = {{"role", "assistant"}, {"content", text}};
  else choice["text"] = text;
  choice["index"] = 0;
  choice["finish_reason"] = std::string(finish_reason);
  choice["logprobs"] = Json();
  Json body = Json::object();
  body["id"] = (chat ? "chatcmpl-" : "cmpl-") + id;
  body["object"] = chat ? "chat.completion" : "text_completion";
  body["created"] = created;
  body["model"] = model;
  body["choices"] = Json::array({std::move(choice)});
  Json usage = Json::object();
  usage["prompt_tokens"] = prompt_tokens;
  usage["completion_tokens"] = completion_tokens;
  usage["total_tokens"] = prompt_tokens + completion_tokens;
  body["usage"] = std::move(usage);
  return body;
}

std::string MakeSseChunk(const std::string& id, const std::string& model,
                         std::int64_t created, std::string_view delta,
                         std::optional<std::string_view> finish_reason, bool chat, bool initial_role) {
  if (chat) {
    Json part = {{"content", delta}};
    if (initial_role) part["role"] = "assistant";
    Json event = {{"id", "chatcmpl-" + id}, {"object", "chat.completion.chunk"},
                  {"created", created}, {"model", model},
                  {"choices", Json::array({{{"index", 0}, {"delta", part},
                    {"finish_reason", finish_reason ? Json(*finish_reason) : Json()}}})}};
    return "data: " + event.dump(-1, ' ', false, Json::error_handler_t::replace) + "\n\n";
  }
  std::string event;
  event.reserve(160 + 2 * delta.size());
  event += "data: {\"id\":\"cmpl-";
  event += id;
  event += "\",\"object\":\"text_completion_chunk\",\"created\":";
  event += std::to_string(created);
  event += ",\"model\":";
  event += SafeDump(model);
  event += ",\"choices\":[{\"index\":0,\"text\":";
  event += SafeDump(delta);
  event += ",\"finish_reason\":";
  if (finish_reason.has_value()) {
    event += SafeDump(*finish_reason);
  } else {
    event += "null";
  }
  event += "}]}\n\n";
  return event;
}

std::string MakeSseUsageChunk(const std::string& id, const std::string& model,
                              std::int64_t created, int prompt_tokens,
                              int completion_tokens, bool chat) {
  if (chat) {
    Json event = {{"id", "chatcmpl-" + id}, {"object", "chat.completion.chunk"},
                  {"created", created}, {"model", model}, {"choices", Json::array()},
                  {"usage", {{"prompt_tokens", prompt_tokens}, {"completion_tokens", completion_tokens},
                              {"total_tokens", prompt_tokens + completion_tokens}}}};
    return "data: " + event.dump() + "\n\n";
  }
  std::string event;
  event.reserve(224);
  event += "data: {\"id\":\"cmpl-";
  event += id;
  event += "\",\"object\":\"text_completion_chunk\",\"created\":";
  event += std::to_string(created);
  event += ",\"model\":";
  event += SafeDump(model);
  event += ",\"choices\":[],\"usage\":{\"prompt_tokens\":";
  event += std::to_string(prompt_tokens);
  event += ",\"completion_tokens\":";
  event += std::to_string(completion_tokens);
  event += ",\"total_tokens\":";
  event += std::to_string(prompt_tokens + completion_tokens);
  event += "}}\n\n";
  return event;
}

Json ApiError::ToJson() const {
  Json error = Json::object();
  error["message"] = message;
  error["type"] = type;
  error["param"] = param.empty() ? Json() : Json(param);
  error["code"] = code.empty() ? Json() : Json(code);
  Json body = Json::object();
  body["error"] = std::move(error);
  return body;
}

Json MakeStreamErrorFrame(const ApiError& error) {
  // Same envelope as unary errors; the HTTP status is already committed
  // when this frame is used, so it is not repeated on the wire.
  return error.ToJson();
}

ApiError MapStatusToApiError(const Status& status) {
  const std::string message(status.message());
  switch (status.code()) {
    case absl::StatusCode::kInvalidArgument:
      return InvalidRequest(message.empty() ? "invalid request" : message);
    case absl::StatusCode::kResourceExhausted:
      return ServerOverloaded(std::chrono::seconds(1));
    case absl::StatusCode::kUnavailable:
      return EngineUnavailable(message.empty() ? "the engine is unavailable" : message);
    case absl::StatusCode::kFailedPrecondition:
      return ShuttingDown();
    case absl::StatusCode::kUnimplemented:
      return NotImplemented(message.empty() ? "not implemented" : message);
    default:
      return Internal("request failed");
  }
}

}  // namespace inferx
