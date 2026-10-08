/// \file
/// \brief Explicit builtin architecture registration. Kept separate from
/// the model implementations: adding an architecture means adding one
/// RegisterArchitecture line here, not extending a global configuration
/// variant. Identities must stay unambiguous.

#include "inferx/models/registry.h"

#include <cstdio>
#include <cstdlib>

#include "inferx/core/logging.h"
#include "inferx/models/architectures/deepseek_v3/model.h"
#include "inferx/models/architectures/gemma3/model.h"
#include "inferx/models/architectures/gpt_oss/model.h"
#include "inferx/models/architectures/llama/model.h"
#include "inferx/models/architectures/mistral/model.h"
#include "inferx/models/architectures/mixtral/model.h"
#include "inferx/models/architectures/qwen2/model.h"
#include "inferx/models/architectures/qwen3/model.h"
#include "inferx/models/architectures/qwen3_moe/model.h"
#include "inferx/models/architectures/qwen3_next/model.h"

namespace inferx::models {

Status ArchitectureRegistry::Register(ArchitectureEntry entry) {
  if (entry.build == nullptr) return InvalidArgumentError("architecture entry has no factory");
  if (entry.architecture.empty()) {
    return InvalidArgumentError("architecture entry needs an architecture name");
  }
  for (const auto& existing : entries_) {
    if ((!entry.architecture.empty() && entry.architecture == existing.architecture) ||
        (!entry.model_type.empty() && !existing.model_type.empty() &&
         entry.model_type == existing.model_type)) {
      return InvalidArgumentError("architecture identity conflict between ",
                                  entry.architecture, " and ", existing.architecture);
    }
  }
  entries_.push_back(std::move(entry));
  return OkStatus();
}

StatusOr<const ArchitectureEntry*> ArchitectureRegistry::Resolve(
    std::string_view model_type, std::string_view architecture) const {
  const ArchitectureEntry* by_type = nullptr;
  const ArchitectureEntry* by_arch = nullptr;
  for (const auto& entry : entries_) {
    if (!model_type.empty() && !entry.model_type.empty() && model_type == entry.model_type) {
      by_type = &entry;
    }
    if (!architecture.empty() && architecture == entry.architecture) {
      by_arch = &entry;
    }
  }
  if (by_type != nullptr && by_arch != nullptr && by_type != by_arch) {
    return InvalidArgumentError("model_type and architectures select different architectures");
  }
  if (by_type != nullptr) return by_type;
  if (by_arch != nullptr) return by_arch;
  std::string supported;
  for (const auto& entry : entries_) {
    if (!supported.empty()) supported += ", ";
    supported += entry.architecture;
    supported += " (";
    supported += entry.model_type;
    supported += ")";
  }
  return UnimplementedError("unsupported model architecture: ", model_type, " (",
                            architecture, "); supported: ", supported);
}

void FatalRegistryError(std::string_view architecture, const Status& status) {
  INFERX_LOG(ERROR) << "builtin architecture registration failed for " << architecture
                    << ": " << status;
  std::abort();
}

const ArchitectureRegistry& ArchitectureRegistry::Builtins() {
  static const ArchitectureRegistry* registry = [] {
    auto* registry = new ArchitectureRegistry();
    RegisterArchitecture<llama::LlamaForCausalLM>(*registry, "LlamaForCausalLM", "llama");
    RegisterArchitecture<qwen2::Qwen2ForCausalLM>(*registry, "Qwen2ForCausalLM", "qwen2");
    RegisterArchitecture<qwen3::Qwen3ForCausalLM>(*registry, "Qwen3ForCausalLM", "qwen3");
    RegisterArchitecture<mistral::MistralForCausalLM>(*registry, "MistralForCausalLM",
                                                      "mistral");
    RegisterArchitecture<gemma3::Gemma3ForCausalLM>(*registry, "Gemma3ForCausalLM",
                                                    "gemma3_text");
    RegisterArchitecture<mixtral::MixtralForCausalLM>(*registry, "MixtralForCausalLM",
                                                      "mixtral");
    RegisterArchitecture<qwen3_moe::Qwen3MoeForCausalLM>(*registry, "Qwen3MoeForCausalLM",
                                                        "qwen3_moe");
    RegisterArchitecture<qwen3_next::Qwen3NextForCausalLM>(*registry, "Qwen3NextForCausalLM",
                                                          "qwen3_next");
    RegisterArchitecture<gpt_oss::GptOssForCausalLM>(*registry, "GptOssForCausalLM",
                                                     "gpt_oss");
    RegisterArchitecture<deepseek_v3::DeepseekV3ForCausalLM>(*registry,
                                                             "DeepseekV3ForCausalLM",
                                                             "deepseek_v3");
    return registry;
  }();
  return *registry;
}

}  // namespace inferx::models
