// Implementation of the OpenAI-compatible API schemas (see api.h).
#include "inferx/server/api.h"

#include <cmath>
#include <stdexcept>

namespace inferx::server {
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
  if (!j.contains("prompt") || !j["prompt"].is_string()) {
    return Failure(400, "field 'prompt' is required and must be a string", "prompt");
  }
  out.request.prompt = j["prompt"].get<std::string>();

  // Everything else merges over the server's defaults.
  sampling::SamplingParams p = defaults;
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
  // Accepted and ignored: usage is always emitted on the final event.
  if (j.contains("stream_options") && !j["stream_options"].is_object()) {
    return Failure(400, "field 'stream_options' must be an object", "stream_options");
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

std::string_view FinishReasonName(FinishReason reason) {
  switch (reason) {
    case FinishReason::kLengthCapped:
      return "length";
    case FinishReason::kStopped:
    default:
      return "stop";
  }
}

Json MakeError(int status, std::string message, std::string param, std::string code) {
  const char* type = "invalid_request_error";
  if (status == 503 || status == 500) type = "server_error";
  if (status == 413) type = "request_too_large";
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
                            int completion_tokens) {
  Json choice = Json::object();
  choice["text"] = text;
  choice["index"] = 0;
  choice["finish_reason"] = std::string(finish_reason);
  choice["logprobs"] = Json();
  Json body = Json::object();
  body["id"] = "cmpl-" + id;
  body["object"] = "text_completion";
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
                         std::optional<std::string_view> finish_reason) {
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
                              int completion_tokens) {
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

}  // namespace inferx::server
