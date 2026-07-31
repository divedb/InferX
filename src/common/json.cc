#include "inferx/common/json.h"

#include <cmath>
#include <cstdlib>
#include <utility>

#include "absl/strings/str_cat.h"

namespace inferx {
namespace {

class Parser {
 public:
  explicit Parser(std::string_view text) : text_(text) {}

  StatusOr<JsonValue> ParseDocument() {
    SkipWhitespace();
    INFERX_ASSIGN_OR_RETURN(JsonValue v, ParseValue(0));
    SkipWhitespace();

    if (pos_ != text_.size()) return Fail("trailing content after JSON value");

    return v;
  }

 private:
  // Guards against a hostile or corrupt file recursing the parser into a stack
  // overflow. Real headers nest three deep at most.
  static constexpr int kMaxDepth = 64;

  Status Fail(std::string_view what) const {
    return InvalidArgumentError("JSON at byte ", pos_, ": ", what);
  }

  void SkipWhitespace() {
    while (pos_ < text_.size()) {
      const char c = text_[pos_];
      if (c == ' ' || c == '\t' || c == '\n' || c == '\r') {
        ++pos_;
      } else {
        break;
      }
    }
  }

  bool Consume(char c) {
    if (pos_ < text_.size() && text_[pos_] == c) {
      ++pos_;
      return true;
    }
    return false;
  }

  bool ConsumeLiteral(std::string_view lit) {
    if (text_.substr(pos_, lit.size()) == lit) {
      pos_ += lit.size();
      return true;
    }
    return false;
  }

  StatusOr<JsonValue> ParseValue(int depth) {
    if (depth > kMaxDepth) return Fail("nesting too deep");
    if (pos_ >= text_.size()) return Fail("unexpected end of input");

    switch (text_[pos_]) {
      case '{': return ParseObject(depth);
      case '[': return ParseArray(depth);
      case '"': {
        INFERX_ASSIGN_OR_RETURN(std::string s, ParseString());
        return JsonValue::String(std::move(s));
      }
      case 't':
        if (ConsumeLiteral("true")) return JsonValue::Bool(true);
        return Fail("invalid literal");
      case 'f':
        if (ConsumeLiteral("false")) return JsonValue::Bool(false);
        return Fail("invalid literal");
      case 'n':
        if (ConsumeLiteral("null")) return JsonValue::Null();
        return Fail("invalid literal");
      default:
        return ParseNumber();
    }
  }

  StatusOr<JsonValue> ParseObject(int depth) {
    if (!Consume('{')) return Fail("expected '{'");

    std::map<std::string, JsonValue> out;

    SkipWhitespace();
    if (Consume('}')) return JsonValue::Object(std::move(out));

    for (;;) {
      SkipWhitespace();

      INFERX_ASSIGN_OR_RETURN(std::string key, ParseString());

      SkipWhitespace();
      if (!Consume(':')) return Fail("expected ':' after object key");

      SkipWhitespace();
      INFERX_ASSIGN_OR_RETURN(JsonValue value, ParseValue(depth + 1));

      // Rejected rather than last-wins: a duplicate key in a checkpoint header
      // means the file is not what we think it is, and picking one silently is
      // how a model loads with the wrong weights.
      if (!out.emplace(std::move(key), std::move(value)).second) {
        return Fail("duplicate object key");
      }

      SkipWhitespace();
      if (Consume(',')) continue;
      if (Consume('}')) break;

      return Fail("expected ',' or '}' in object");
    }

    return JsonValue::Object(std::move(out));
  }

  StatusOr<JsonValue> ParseArray(int depth) {
    if (!Consume('[')) return Fail("expected '['");

    std::vector<JsonValue> out;

    SkipWhitespace();
    if (Consume(']')) return JsonValue::Array(std::move(out));

    for (;;) {
      SkipWhitespace();
      INFERX_ASSIGN_OR_RETURN(JsonValue v, ParseValue(depth + 1));
      out.push_back(std::move(v));

      SkipWhitespace();
      if (Consume(',')) continue;
      if (Consume(']')) break;

      return Fail("expected ',' or ']' in array");
    }

    return JsonValue::Array(std::move(out));
  }

  StatusOr<std::string> ParseString() {
    if (!Consume('"')) return Fail("expected '\"'");

    std::string out;

    while (pos_ < text_.size()) {
      const char c = text_[pos_++];

      if (c == '"') return out;

      if (c != '\\') {
        out.push_back(c);
        continue;
      }

      if (pos_ >= text_.size()) return Fail("unterminated escape");

      switch (const char esc = text_[pos_++]) {
        case '"':  out.push_back('"');  break;
        case '\\': out.push_back('\\'); break;
        case '/':  out.push_back('/');  break;
        case 'b':  out.push_back('\b'); break;
        case 'f':  out.push_back('\f'); break;
        case 'n':  out.push_back('\n'); break;
        case 'r':  out.push_back('\r'); break;
        case 't':  out.push_back('\t'); break;
        case 'u': {
          if (pos_ + 4 > text_.size()) return Fail("truncated \\u escape");

          int code = 0;
          for (int i = 0; i < 4; ++i) {
            const char h = text_[pos_ + i];
            code *= 16;
            if (h >= '0' && h <= '9') code += h - '0';
            else if (h >= 'a' && h <= 'f') code += h - 'a' + 10;
            else if (h >= 'A' && h <= 'F') code += h - 'A' + 10;
            else return Fail("bad hex digit in \\u escape");
          }
          pos_ += 4;

          // Rejected rather than mangled. Tensor names and config keys are
          // ASCII; anything else means this is not the file we expect, and a
          // half-decoded name would fail a lookup much later and much less
          // legibly.
          if (code > 0x7F) return Fail("non-ASCII \\u escape is unsupported");

          out.push_back(static_cast<char>(code));
          break;
        }
        default:
          (void)esc;
          return Fail("invalid escape character");
      }
    }

    return Fail("unterminated string");
  }

  StatusOr<JsonValue> ParseNumber() {
    const size_t start = pos_;

    if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) ++pos_;

    bool any_digits = false;
    while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
      ++pos_;
      any_digits = true;
    }

    if (pos_ < text_.size() && text_[pos_] == '.') {
      ++pos_;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
        any_digits = true;
      }
    }

    if (!any_digits) return Fail("expected a number");

    if (pos_ < text_.size() && (text_[pos_] == 'e' || text_[pos_] == 'E')) {
      ++pos_;
      if (pos_ < text_.size() && (text_[pos_] == '-' || text_[pos_] == '+')) {
        ++pos_;
      }
      bool exp_digits = false;
      while (pos_ < text_.size() && text_[pos_] >= '0' && text_[pos_] <= '9') {
        ++pos_;
        exp_digits = true;
      }
      if (!exp_digits) return Fail("exponent has no digits");
    }

    // strtod over the substring: the text is not null-terminated, so it is
    // copied into a small buffer first. Numbers here are short (dimensions,
    // epsilons, rope theta).
    const std::string token(text_.substr(start, pos_ - start));

    errno = 0;
    char* end = nullptr;
    const double v = std::strtod(token.c_str(), &end);

    if (end != token.c_str() + token.size()) return Fail("malformed number");
    if (!std::isfinite(v)) return Fail("number is not finite");

    return JsonValue::Number(v);
  }

  std::string_view text_;
  size_t pos_ = 0;
};

}  // namespace

JsonValue JsonValue::Bool(bool v) {
  JsonValue j;
  j.kind_ = Kind::kBool;
  j.bool_ = v;
  return j;
}

JsonValue JsonValue::Number(double v) {
  JsonValue j;
  j.kind_ = Kind::kNumber;
  j.number_ = v;
  return j;
}

JsonValue JsonValue::String(std::string v) {
  JsonValue j;
  j.kind_ = Kind::kString;
  j.string_ = std::move(v);
  return j;
}

JsonValue JsonValue::Array(std::vector<JsonValue> v) {
  JsonValue j;
  j.kind_ = Kind::kArray;
  j.array_ = std::move(v);
  return j;
}

JsonValue JsonValue::Object(std::map<std::string, JsonValue> v) {
  JsonValue j;
  j.kind_ = Kind::kObject;
  j.object_ = std::move(v);
  return j;
}

const char* JsonValue::KindName() const {
  switch (kind_) {
    case Kind::kNull:   return "null";
    case Kind::kBool:   return "bool";
    case Kind::kNumber: return "number";
    case Kind::kString: return "string";
    case Kind::kArray:  return "array";
    case Kind::kObject: return "object";
  }
  return "?";
}

const JsonValue* JsonValue::Find(std::string_view key) const {
  if (kind_ != Kind::kObject) return nullptr;

  const auto it = object_.find(std::string(key));

  return it == object_.end() ? nullptr : &it->second;
}

StatusOr<bool> JsonValue::AsBool() const {
  if (kind_ != Kind::kBool) {
    return InvalidArgumentError("expected bool, got ", KindName());
  }
  return bool_;
}

StatusOr<double> JsonValue::AsDouble() const {
  if (kind_ != Kind::kNumber) {
    return InvalidArgumentError("expected number, got ", KindName());
  }
  return number_;
}

StatusOr<int64_t> JsonValue::AsInt() const {
  if (kind_ != Kind::kNumber) {
    return InvalidArgumentError("expected number, got ", KindName());
  }

  // Checked rather than truncated: a dimension that arrives as 2048.5 means the
  // file is wrong, and rounding it produces a model that loads and misbehaves.
  if (number_ != std::floor(number_)) {
    return InvalidArgumentError("expected an integer, got ", number_);
  }

  if (number_ < -9.2e18 || number_ > 9.2e18) {
    return InvalidArgumentError("integer out of range: ", number_);
  }

  return static_cast<int64_t>(number_);
}

StatusOr<std::string_view> JsonValue::AsString() const {
  if (kind_ != Kind::kString) {
    return InvalidArgumentError("expected string, got ", KindName());
  }
  return std::string_view(string_);
}

StatusOr<const std::vector<JsonValue>*> JsonValue::AsArray() const {
  if (kind_ != Kind::kArray) {
    return InvalidArgumentError("expected array, got ", KindName());
  }
  return &array_;
}

StatusOr<const std::map<std::string, JsonValue>*> JsonValue::AsObject() const {
  if (kind_ != Kind::kObject) {
    return InvalidArgumentError("expected object, got ", KindName());
  }
  return &object_;
}

StatusOr<int64_t> JsonValue::RequiredInt(std::string_view key) const {
  const JsonValue* v = Find(key);
  if (v == nullptr) return InvalidArgumentError("missing required key '", key, "'");

  auto r = v->AsInt();
  if (!r.ok()) return InvalidArgumentError("key '", key, "': ", r.status().message());

  return r;
}

StatusOr<double> JsonValue::RequiredDouble(std::string_view key) const {
  const JsonValue* v = Find(key);
  if (v == nullptr) return InvalidArgumentError("missing required key '", key, "'");

  auto r = v->AsDouble();
  if (!r.ok()) return InvalidArgumentError("key '", key, "': ", r.status().message());

  return r;
}

StatusOr<std::string_view> JsonValue::RequiredString(std::string_view key) const {
  const JsonValue* v = Find(key);
  if (v == nullptr) return InvalidArgumentError("missing required key '", key, "'");

  auto r = v->AsString();
  if (!r.ok()) return InvalidArgumentError("key '", key, "': ", r.status().message());

  return r;
}

StatusOr<int64_t> JsonValue::OptionalInt(std::string_view key,
                                         int64_t fallback) const {
  const JsonValue* v = Find(key);
  if (v == nullptr || v->IsNull()) return fallback;

  auto r = v->AsInt();
  if (!r.ok()) return InvalidArgumentError("key '", key, "': ", r.status().message());

  return r;
}

StatusOr<bool> JsonValue::OptionalBool(std::string_view key,
                                       bool fallback) const {
  const JsonValue* v = Find(key);
  if (v == nullptr || v->IsNull()) return fallback;

  auto r = v->AsBool();
  if (!r.ok()) return InvalidArgumentError("key '", key, "': ", r.status().message());

  return r;
}

StatusOr<JsonValue> ParseJson(std::string_view text) {
  return Parser(text).ParseDocument();
}

}  // namespace inferx
