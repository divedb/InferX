#include "inferx/config/load_config.h"

#include <array>
#include <string>
#include <utility>

namespace inferx {

namespace {

constexpr std::array<std::pair<LoadFormat, const char*>, 5> kLoadFormats = {{
    {LoadFormat::kAuto, "auto"},
    {LoadFormat::kSafetensors, "safetensors"},
    {LoadFormat::kPt, "pt"},
    {LoadFormat::kSharded, "sharded"},
    {LoadFormat::kDummy, "dummy"},
}};

}  // namespace

const char* ToString(LoadFormat format) {
  for (const auto& [value, name] : kLoadFormats) {
    if (value == format) return name;
  }
  return "unknown";
}

Status ParseLoadFormat(const std::string& text, LoadFormat* out) {
  for (const auto& [value, name] : kLoadFormats) {
    if (text == name) {
      *out = value;
      return Status::OK();
    }
  }
  return Status::InvalidArgument("unknown load format '" + text +
                                 "'; expected one of: auto, safetensors, pt, sharded, dummy");
}

Status LoadConfig::Validate() const {
  if (pt_load_map_location.empty()) {
    return Status::InvalidArgument("LoadConfig: pt_load_map_location must not be empty");
  }
  for (const std::string& pattern : ignore_patterns) {
    if (pattern.empty()) {
      return Status::InvalidArgument(
          "LoadConfig: ignore_patterns must not contain empty entries");
    }
  }
  return Status::OK();
}

}  // namespace inferx