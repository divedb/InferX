#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/status.h"

namespace inferx {

/// \brief A parsed JSON value.
///
/// Deliberately small and slow. It reads two things -- safetensors headers and
/// `config.json` -- both machine-generated, both read once at startup, neither
/// on any hot path. ARCHITECTURE.md §9 puts simdjson on the request path at M4
/// for the case where parsing speed is load-bearing; pulling it in now to read
/// a 30 KB header once would be a dependency bought for nothing.
///
/// What it does have to be is *strict*. A loader that silently accepts
/// malformed input produces a model with quietly wrong weights, which is far
/// more expensive to debug than a parse error at startup. Duplicate keys,
/// trailing content, unterminated strings and out-of-range numbers are all
/// errors rather than best-effort guesses.
///
/// Not supported, because neither input uses them: `\u` escapes beyond the
/// ASCII range are rejected rather than mis-decoded.
class JsonValue {
 public:
  enum class Kind { kNull, kBool, kNumber, kString, kArray, kObject };

  JsonValue() = default;

  static JsonValue Null() { return JsonValue(); }
  static JsonValue Bool(bool v);
  static JsonValue Number(double v);
  static JsonValue String(std::string v);
  static JsonValue Array(std::vector<JsonValue> v);
  static JsonValue Object(std::map<std::string, JsonValue> v);

  Kind kind() const { return kind_; }
  bool IsNull() const { return kind_ == Kind::kNull; }
  bool IsObject() const { return kind_ == Kind::kObject; }
  bool IsArray() const { return kind_ == Kind::kArray; }

  /// \brief Looks up `key`, or nullptr if absent or this is not an object.
  const JsonValue* Find(std::string_view key) const;

  /// \brief Typed accessors. Each fails rather than coercing: a string "4" is
  ///        not an integer, because a config that says so is a config we do not
  ///        understand.
  StatusOr<bool> AsBool() const;
  StatusOr<double> AsDouble() const;
  StatusOr<int64_t> AsInt() const;
  StatusOr<std::string_view> AsString() const;
  StatusOr<const std::vector<JsonValue>*> AsArray() const;
  StatusOr<const std::map<std::string, JsonValue>*> AsObject() const;

  /// \brief Required-field lookup with the field name in any error.
  StatusOr<int64_t> RequiredInt(std::string_view key) const;
  StatusOr<double> RequiredDouble(std::string_view key) const;
  StatusOr<std::string_view> RequiredString(std::string_view key) const;

  /// \brief Optional field with a fallback. Absent is fine; present-but-wrong
  ///        -type is still an error.
  StatusOr<int64_t> OptionalInt(std::string_view key, int64_t fallback) const;
  StatusOr<bool> OptionalBool(std::string_view key, bool fallback) const;

  const char* KindName() const;

 private:
  Kind kind_ = Kind::kNull;
  bool bool_ = false;
  double number_ = 0.0;
  std::string string_;
  std::vector<JsonValue> array_;
  std::map<std::string, JsonValue> object_;
};

/// \brief Parses `text` as a complete JSON document.
///
/// \param text The document. Must be entirely consumed; trailing non-whitespace
///             is an error.
/// \return     The value, or InvalidArgument naming the byte offset.
StatusOr<JsonValue> ParseJson(std::string_view text);

/// \brief Appends `text` to `out` as a quoted JSON string literal.
///
/// The API layer builds responses by concatenation rather than by assembling a
/// `JsonValue` tree, because the shapes are fixed and known. That is fine for
/// the scaffolding and lethal for the payload: model output is arbitrary text
/// that routinely contains quotes, backslashes and newlines, and one unescaped
/// character turns a completion into a parse error at the client -- or, worse,
/// lets generated text close the string and forge the rest of the object.
///
/// Escapes what RFC 8259 requires and nothing more: quote, backslash, and the
/// C0 controls, the last as `\uXXXX` where there is no short form. Bytes above
/// 0x7F are passed through, so valid UTF-8 in gives valid UTF-8 out.
void AppendJsonString(std::string_view text, std::string* out);

}  // namespace inferx
