#include "inferx/util/config_view.h"

#include <cmath>
#include <nlohmann/json.hpp>

#include "absl/strings/str_cat.h"

namespace inferx {
namespace {

using absl::StrCat;
using nlohmann::json;

Status WrongType(const std::string& path, const char* want, const json& v) {
  return InvalidArgumentError(
      StrCat("config key '", path, "': expected ", want, ", got ", v.type_name()));
}

Status Missing(const std::string& path) {
  return InvalidArgumentError(StrCat("config key '", path, "' is required"));
}

StatusOr<int64_t> AsInt(const std::string& path, const json& v) {
  if (v.is_number_integer()) return v.get<int64_t>();

  // Some checkpoints write integral values as 128.0.
  if (v.is_number_float()) {
    const double d = v.get<double>();

    // TODO(gc): This is a bit of a hack. We should probably have a more robust way to handle
    //           this.
    if (std::floor(d) == d && std::fabs(d) < 9e15) return static_cast<int64_t>(d);
  }

  return WrongType(path, "integer", v);
}

StatusOr<double> AsDouble(const std::string& path, const json& v) {
  if (v.is_number()) return v.get<double>();

  return WrongType(path, "number", v);
}

StatusOr<bool> AsBool(const std::string& path, const json& v) {
  if (!v.is_boolean()) return WrongType(path, "boolean", v);

  return v.get<bool>();
}

StatusOr<std::string> AsString(const std::string& path, const json& v) {
  if (!v.is_string()) return WrongType(path, "string", v);

  return v.get<std::string>();
}

}  // namespace

template <typename T, typename Convert>
StatusOr<T> ConfigView::RequireAs(std::string_view key, Convert convert) const {
  const json* v = Find(key);

  if (v == nullptr) return Missing(PathOf(key));

  return convert(PathOf(key), *v);
}

template <typename T, typename Convert>
StatusOr<T> ConfigView::GetAs(std::string_view key, T default_value, Convert convert) const {
  const json* v = Find(key);

  if (v == nullptr) return default_value;

  return convert(PathOf(key), *v);
}

std::string ConfigView::PathOf(std::string_view key) const {
  return path_.empty() ? std::string(key) : StrCat(path_, ".", key);
}

const json* ConfigView::Find(std::string_view key) const {
  if (node_ == nullptr || !node_->is_object()) return nullptr;

  auto it = node_->find(std::string(key));

  if (it == node_->end() || it->is_null()) return nullptr;

  return &*it;
}

bool ConfigView::Has(std::string_view key) const { return Find(key) != nullptr; }

StatusOr<int64_t> ConfigView::RequireInt(std::string_view key) const {
  return RequireAs<int64_t>(key, AsInt);
}

StatusOr<int64_t> ConfigView::Int(std::string_view key, int64_t default_value) const {
  return GetAs(key, default_value, AsInt);
}

StatusOr<double> ConfigView::RequireFloat(std::string_view key) const {
  return RequireAs<double>(key, AsDouble);
}

StatusOr<double> ConfigView::Float(std::string_view key, double default_value) const {
  return GetAs(key, default_value, AsDouble);
}

StatusOr<std::optional<double>> ConfigView::OptFloat(std::string_view key) const {
  const json* v = Find(key);

  if (v == nullptr) return std::optional<double>{};

  auto d = AsDouble(PathOf(key), *v);

  if (!d.ok()) return d.status();

  return std::optional<double>{*d};
}

StatusOr<bool> ConfigView::Bool(std::string_view key, bool default_value) const {
  return GetAs(key, default_value, AsBool);
}

StatusOr<std::string> ConfigView::RequireStr(std::string_view key) const {
  return RequireAs<std::string>(key, AsString);
}

StatusOr<std::string> ConfigView::Str(std::string_view key, std::string default_value) const {
  return GetAs(key, std::move(default_value), AsString);
}

StatusOr<std::vector<int64_t>> ConfigView::IntList(std::string_view key) const {
  std::vector<int64_t> out;
  const json* v = Find(key);

  if (v == nullptr) return out;

  const std::string path = PathOf(key);

  if (v->is_array()) {
    for (size_t i = 0; i < v->size(); ++i) {
      auto x = AsInt(StrCat(path, "[", i, "]"), (*v)[i]);

      if (!x.ok()) return x.status();

      out.push_back(*x);
    }

    return out;
  }

  auto x = AsInt(path, *v);

  if (!x.ok()) return x.status();

  out.push_back(*x);

  return out;
}

StatusOr<std::vector<std::string>> ConfigView::StrList(std::string_view key) const {
  std::vector<std::string> out;
  const json* v = Find(key);

  if (v == nullptr) return out;
  if (!v->is_array()) return WrongType(PathOf(key), "array of strings", *v);

  const std::string path = PathOf(key);
  for (size_t i = 0; i < v->size(); ++i) {
    auto x = AsString(StrCat(path, "[", i, "]"), (*v)[i]);
    if (!x.ok()) return x.status();
    out.push_back(std::move(*x));
  }
  return out;
}

StatusOr<std::optional<ConfigView>> ConfigView::Sub(std::string_view key) const {
  const json* v = Find(key);

  if (v == nullptr) return std::optional<ConfigView>{};

  if (!v->is_object()) return WrongType(PathOf(key), "object", *v);

  return std::optional<ConfigView>{ConfigView(v, PathOf(key))};
}

Status ConfigView::RejectIfPresent(std::string_view key, std::string_view why) const {
  if (Find(key) == nullptr) return OkStatus();

  return UnimplementedError(
      StrCat("config key '", PathOf(key), "' is set: ", why, " is not supported"));
}

}  // namespace inferx
