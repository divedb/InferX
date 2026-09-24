#include "inferx/models/model_registry.h"

#include <vector>

#include "inferx/models/causal/causal_lm.h"
#include "models/causal/config_parser.h"

namespace inferx {
namespace {

/// Tier-1 rows: standard dense decoders, knobs only. Adding a Llama-shaped
/// family is one row here and zero new files.
const std::vector<Family>& StandardFamilies() {
  static const std::vector<Family> families = {
      Family{.model_type = "llama", .architecture = "LlamaForCausalLM"},
  };
  return families;
}

const Family* Lookup(const CheckpointConfig& config) {
  const auto matches = [&](const Family& family) {
    return config.model_type == family.model_type ||
           (!config.architectures.empty() && config.architectures == family.architecture);
  };
  for (const auto& family : StandardFamilies()) {
    if (matches(family)) return &family;
  }
  for (const auto& family : RegisteredFamilies()) {
    if (matches(family)) return &family;
  }
  return nullptr;
}

StatusOr<causal::DecoderConfig> TranslateFamilyConfig(const Family& family,
                                                      std::string_view config_json) {
  INFERX_ASSIGN_OR_RETURN(auto json, causal::ParseConfigJson(config_json));
  INFERX_ASSIGN_OR_RETURN(auto config,
                          family.translate != nullptr
                              ? family.translate(json)
                              : causal::AttentionDecoderConfig(json, family.qk_norm,
                                                               family.plus_one_norm));
  INFERX_RETURN_IF_ERROR(config.Validate());
  return config;
}

}  // namespace

StatusOr<const Family*> ResolveFamily(const CheckpointConfig& config) {
  if (const Family* family = Lookup(config)) return family;
  return UnimplementedError("unsupported model architecture: ", config.model_type, " (",
                            config.architectures, ")");
}

StatusOr<std::unique_ptr<Model>> BuildFamily(const Family& family,
                                             models::LoadedCheckpoint& checkpoint,
                                             DeviceId device, int max_tokens, int max_seqs,
                                             const ParallelConfig& parallel) {
  if (family.build != nullptr) {
    INFERX_ASSIGN_OR_RETURN(auto json, causal::ParseConfigJson(checkpoint.config_json));
    return family.build(json, checkpoint, device, max_tokens, max_seqs, parallel);
  }
  INFERX_ASSIGN_OR_RETURN(auto config, TranslateFamilyConfig(family, checkpoint.config_json));
  return causal::BuildCausalLM(checkpoint, std::move(config), family.layout, device,
                               max_tokens, max_seqs, parallel);
}

StatusOr<causal::DecoderConfig> TranslateFamilyConfig(const CheckpointConfig& identity,
                                                      std::string_view config_json) {
  INFERX_ASSIGN_OR_RETURN(const Family* family, ResolveFamily(identity));
  return TranslateFamilyConfig(*family, config_json);
}

StatusOr<std::unique_ptr<Model>> BuildModel(models::LoadedCheckpoint& checkpoint,
                                            DeviceId device, int max_tokens, int max_seqs,
                                            const ParallelConfig& parallel) {
  INFERX_ASSIGN_OR_RETURN(const Family* family, ResolveFamily(checkpoint.config));
  return BuildFamily(*family, checkpoint, device, max_tokens, max_seqs, parallel);
}

}  // namespace inferx
