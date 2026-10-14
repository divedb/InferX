#pragma once

#include <cstdint>
#include <nlohmann/json_fwd.hpp>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/status.h"

namespace inferx {

/// \brief Read-only, type-checked view over one JSON object.
///
/// Every getter reports the full dotted key path on failure
/// ("text_config.rope_scaling.factor"), so a bad config.json points at the
/// exact key. A key that is absent or JSON null counts as "not present".
/// Optional getters still fail on a wrong type: only absence falls back to the
/// default, so a malformed value never turns into a silent default.
///
/// A view borrows its JSON node; it must not outlive the owning HfConfig.
class ConfigView {
 public:
  ConfigView() = default;

  /// \brief Construct a view over a JSON node.
  ///
  /// \param node The JSON node to view.
  /// \param path The path of the node.
  ConfigView(const nlohmann::json* node, std::string path)
      : node_(node), path_(std::move(path)) {}

  const std::string& Path() const { return path_; }

  /// \brief True when `key` is present and not null.
  bool Has(std::string_view key) const;

  /// \brief Get an integer value, or an error if absent or not an integer.
  ///
  /// \param key The key to look up.
  /// \return    The integer value of the key, or an error if absent or not an integer.
  StatusOr<int64_t> RequireInt(std::string_view key) const;

  /// \brief Get a float value, or an error if absent or not a float.
  ///
  /// \param key The key to look up.
  /// \return    The float value of the key, or an error if absent or not a float.
  StatusOr<double> RequireFloat(std::string_view key) const;

  /// \brief Get a string value, or an error if absent or not a string.
  ///
  /// \param key The key to look up.
  /// \return    The string value of the key, or an error if absent or not a string.
  StatusOr<std::string> RequireStr(std::string_view key) const;

  /// \brief Get an integer value, or `default_value` if absent.
  ///
  /// \param key           The key to look up.
  /// \param default_value The value to return if the key is absent.
  /// \return              The integer value of the key, or `default_value` if absent.
  StatusOr<int64_t> Int(std::string_view key, int64_t default_value) const;

  /// \brief Get a float value, or `default_value` if absent.
  ///
  /// \param key           The key to look up.
  /// \param default_value The value to return if the key is absent.
  /// \return              The float value of the key, or `default_value` if absent.
  StatusOr<double> Float(std::string_view key, double default_value) const;

  /// \brief Get a boolean value, or `default_value` if absent.
  ///
  /// \param key           The key to look up.
  /// \param default_value The value to return if the key is absent.
  /// \return              The boolean value of the key, or `default_value` if absent.
  StatusOr<bool> Bool(std::string_view key, bool default_value) const;

  /// \brief Get a string value, or `default_value` if absent.
  ///
  /// \param key           The key to look up.
  /// \param default_value The value to return if the key is absent.
  /// \return              The string value of the key, or `default_value` if absent
  StatusOr<std::string> Str(std::string_view key, std::string default_value) const;

  /// \brief Optional float; nullopt when absent.
  StatusOr<std::optional<double>> OptFloat(std::string_view key) const;

  /// \brief Accepts a scalar or an array of integers (e.g. `eos_token_id`).
  ///        Empty when absent.
  ///
  /// \param key The key to look up.
  /// \return    The integer value(s) of the key, or an error if present but not an integer or
  ///            array of integers.
  StatusOr<std::vector<int64_t>> IntList(std::string_view key) const;

  /// \brief Accepts an array of strings (e.g. `layer_types`). Empty when
  ///        absent; errors if present but not a string array.
  ///
  /// \param key The key to look up.
  /// \return    The strings of the key, or an error on a non-string entry.
  StatusOr<std::vector<std::string>> StrList(std::string_view key) const;

  /// \brief Nested object; nullopt when absent, error if present but not an object.
  ///
  /// \param key The key to look up.
  /// \return    A view of the nested object, or nullopt if absent, or an error if present but
  ///            not an object.
  StatusOr<std::optional<ConfigView>> Sub(std::string_view key) const;

  /// \brief Returns Unimplemented when `key` is present, for features that
  ///        change numerics or layout and that the engine does not support.
  ///
  /// \param key The key to look up.
  /// \param why A message to include in the error.
  /// \return    Unimplemented if the key is present, otherwise OK.
  Status RejectIfPresent(std::string_view key, std::string_view why) const;

  /// \brief Get the path of a key.
  ///
  /// \param key The key to look up.
  /// \return    The path of the key.
  std::string PathOf(std::string_view key) const;

 private:
  const nlohmann::json* Find(std::string_view key) const;

  // Shared getter core: Find `key`; on absence return Missing(path) for
  // RequireAs or `default_value` for GetAs; otherwise convert(path, value).
  // `convert` has the shape of the As* helpers in config_view.cc.
  template <typename T, typename Convert>
  StatusOr<T> RequireAs(std::string_view key, Convert convert) const;

  template <typename T, typename Convert>
  StatusOr<T> GetAs(std::string_view key, T default_value, Convert convert) const;

  const nlohmann::json* node_ = nullptr;
  std::string path_;
};

}  // namespace inferx
