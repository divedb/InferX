#include "inferx/server/json_codec.h"

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include <nlohmann/json.hpp>

namespace inferx::server {

bool IsValidUtf8(std::string_view text) {
  const auto* p = reinterpret_cast<const unsigned char*>(text.data());
  const auto* end = p + text.size();
  while (p < end) {
    const unsigned char c = *p;
    if (c < 0x80) {
      ++p;
      continue;
    }
    int continuation = 0;
    unsigned int min_code_point = 0;
    if ((c & 0xE0) == 0xC0) {
      continuation = 1;
      min_code_point = 0x80;
    } else if ((c & 0xF0) == 0xE0) {
      continuation = 2;
      min_code_point = 0x800;
    } else if ((c & 0xF8) == 0xF0) {
      continuation = 3;
      min_code_point = 0x10000;
    } else {
      return false;
    }
    if (end - p < continuation + 1) return false;
    unsigned int code_point = c & (0x3F >> continuation);
    for (int i = 1; i <= continuation; ++i) {
      const unsigned char cc = p[i];
      if ((cc & 0xC0) != 0x80) return false;
      code_point = (code_point << 6) | (cc & 0x3F);
    }
    if (code_point < min_code_point || code_point > 0x10FFFF ||
        (code_point >= 0xD800 && code_point <= 0xDFFF)) {
      return false;
    }
    p += continuation + 1;
  }
  return true;
}

namespace {

using Json = nlohmann::json;

/// \brief SAX consumer building a DOM under HTTP limits: nesting depth,
///        string length, and a total node budget proportional to the body
///        ceiling. Values are assembled bottom-up on a value stack, so no
///        pointers into reallocating containers are held.
class LimitedDomSax {
 public:
  LimitedDomSax(const HttpLimits& limits, std::size_t max_nodes)
      : limits_(limits), max_nodes_(max_nodes) {}

  struct Node {
    Json value;
    std::string key;
  };

  bool ok = true;
  std::string error;
  // Exactly one entry after a successful parse.
  std::vector<Node> values_;

  bool null() { return PushValue(Json(nullptr)); }
  bool boolean(bool b) { return PushValue(Json(b)); }
  bool number_integer(Json::number_integer_t v) { return PushValue(Json(v)); }
  bool number_unsigned(Json::number_unsigned_t v) { return PushValue(Json(v)); }
  bool number_float(Json::number_float_t v, const Json::string_t& raw) {
    (void)raw;
    return PushValue(Json(v));
  }
  bool string(Json::string_t& value) {
    if (!ok) return false;
    if (value.size() > limits_.max_json_string_bytes) {
      return Fail("json string exceeds the configured length limit");
    }
    return PushValue(Json(std::move(value)));
  }
  bool binary(Json::binary_t& value) { return PushValue(Json(std::move(value))); }

  bool start_object(std::size_t) { return StartContainer(/*is_object=*/true); }
  bool key(Json::string_t& value) {
    if (!ok) return false;
    if (value.size() > limits_.max_json_string_bytes) {
      return Fail("json key exceeds the configured length limit");
    }
    pending_key_ = std::move(value);
    return true;
  }
  bool end_object() { return EndContainer(/*is_object=*/true); }
  bool start_array(std::size_t) { return StartContainer(/*is_object=*/false); }
  bool end_array() { return EndContainer(/*is_object=*/false); }

  bool parse_error(std::size_t position, const std::string&,
                   const Json::exception& ex) {
    error = "invalid json at byte " + std::to_string(position) + ": " +
            ex.what();
    ok = false;
    return false;
  }

 private:
  struct Open {
    std::size_t mark;      // values.size() when this container opened
    bool is_object;
    std::string key;       // key this container holds in its parent
  };

  bool Fail(std::string_view detail) {
    error = detail;
    ok = false;
    return false;
  }
  bool CheckNode() {
    if (nodes_ >= max_nodes_) return Fail("json document exceeds the node budget");
    ++nodes_;
    return true;
  }

  bool PushValue(Json value) {
    if (!ok) return false;
    if (!CheckNode()) return false;
    values_.push_back(Node{std::move(value), std::move(pending_key_)});
    pending_key_.clear();
    return true;
  }

  bool StartContainer(bool is_object) {
    if (!ok) return false;
    if (!CheckNode()) return false;
    if (opens_.size() >= limits_.max_json_depth) {
      return Fail("json nesting depth exceeds the configured limit");
    }
    opens_.push_back(Open{values_.size(), is_object, std::move(pending_key_)});
    pending_key_.clear();
    return true;
  }

  bool EndContainer(bool is_object) {
    if (!ok || opens_.empty()) return false;
    const Open open = opens_.back();
    opens_.pop_back();
    Json container = is_object ? Json::object() : Json::array();
    const auto begin = values_.begin() + static_cast<std::ptrdiff_t>(open.mark);
    for (auto it = begin; it != values_.end(); ++it) {
      if (is_object) {
        container[it->key] = std::move(it->value);
      } else {
        container.push_back(std::move(it->value));
      }
    }
    values_.erase(begin, values_.end());
    // The container's own node was counted at StartContainer.
    values_.push_back(Node{std::move(container), std::move(open.key)});
    return true;
  }

  const HttpLimits& limits_;
  std::size_t max_nodes_;
  std::size_t nodes_ = 0;
  std::string pending_key_;
  std::vector<Open> opens_;
};

ApiError JsonError(std::string message) {
  return ApiError{400,                    ApiErrorKind::kInvalidRequest,
                  "invalid_request_error", "invalid_json",
                  std::move(message),     std::nullopt,
                  std::nullopt,           false,
                  {}};
}

}  // namespace

Result<nlohmann::json> ParseJson(std::string_view body,
                                 const HttpLimits& limits,
                                 std::size_t max_body_bytes) {
  if (body.size() > max_body_bytes) {
    return ApiError{413,                    ApiErrorKind::kInvalidRequest,
                    "invalid_request_error", "payload_too_large",
                    "request body exceeds the configured limit",
                    std::nullopt,           std::nullopt,
                    false,                  {}};
  }
  if (!IsValidUtf8(body)) {
    return JsonError("request body is not valid UTF-8");
  }
  const std::size_t max_nodes = max_body_bytes + 1;
  LimitedDomSax sax(limits, max_nodes);
  const bool parsed = Json::sax_parse(body, &sax, Json::input_format_t::json,
                                      /*strict=*/true);
  if (!parsed || !sax.ok || sax.values_.size() != 1) {
    return JsonError(sax.error.empty() ? "malformed json" : sax.error);
  }
  Json root = std::move(sax.values_.front().value);
  return root;
}

Result<std::string> DumpJson(const nlohmann::json& value) {
  try {
    return value.dump();
  } catch (const nlohmann::json::type_error&) {
    return ApiError{500,                    ApiErrorKind::kInternal,
                    "internal_error",        "invalid_json_state",
                    "response serialization failed", std::nullopt,
                    std::nullopt,           false,
                    {}};
  }
}

}  // namespace inferx::server
