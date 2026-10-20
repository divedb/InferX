#include "inferx/config/speculative_config.h"

#include <array>
#include <string>
#include <utility>

namespace inferx {

namespace {

constexpr std::array<std::pair<SpeculativeMethod, const char*>, 7> kMethods = {{
    {SpeculativeMethod::kNone, "none"},
    {SpeculativeMethod::kNgram, "ngram"},
    {SpeculativeMethod::kDraftModel, "draft_model"},
    {SpeculativeMethod::kMedusa, "medusa"},
    {SpeculativeMethod::kEagle, "eagle"},
    {SpeculativeMethod::kEagle3, "eagle3"},
    {SpeculativeMethod::kMtp, "mtp"},
}};

}  // namespace

const char* ToString(SpeculativeMethod method) {
  for (const auto& [value, name] : kMethods) {
    if (value == method) return name;
  }
  return "unknown";
}

Status ParseSpeculativeMethod(const std::string& text, SpeculativeMethod* out) {
  for (const auto& [value, name] : kMethods) {
    if (text == name) {
      *out = value;
      return Status::OK();
    }
  }
  return Status::InvalidArgument(
      "unknown speculative method '" + text +
      "'; expected one of: none, ngram, draft_model, medusa, eagle, eagle3, mtp");
}

bool SpeculativeConfig::RequiresDraftModel() const {
  switch (method) {
    case SpeculativeMethod::kDraftModel:
    case SpeculativeMethod::kMedusa:
    case SpeculativeMethod::kEagle:
    case SpeculativeMethod::kEagle3:
      return true;
    case SpeculativeMethod::kNone:
    case SpeculativeMethod::kNgram:
    case SpeculativeMethod::kMtp:
      return false;
  }
  return false;
}

Status SpeculativeConfig::Validate() const {
  if (!Enabled()) return Status::OK();

  if (num_speculative_tokens < 1) {
    return Status::InvalidArgument("SpeculativeConfig: num_speculative_tokens must be >= 1");
  }
  if (RequiresDraftModel() && model.empty()) {
    return Status::InvalidArgument(std::string("SpeculativeConfig: method '") +
                                   ToString(method) + "' requires a draft model path");
  }
  if (draft_tensor_parallel_size < 0) {
    return Status::InvalidArgument(
        "SpeculativeConfig: draft_tensor_parallel_size must be >= 0");
  }
  if (method == SpeculativeMethod::kNgram &&
      (prompt_lookup_min < 1 || prompt_lookup_max < prompt_lookup_min)) {
    return Status::InvalidArgument(
        "SpeculativeConfig: need 1 <= prompt_lookup_min <= prompt_lookup_max, got min=" +
        std::to_string(prompt_lookup_min) + " max=" + std::to_string(prompt_lookup_max));
  }
  return Status::OK();
}

}  // namespace inferx