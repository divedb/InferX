#include "inferx/models/model_registry.h"

#include <array>

#include "inferx/models/causal/config_parser.h"
#include "inferx/models/families/families.h"

namespace inferx {
namespace {

const auto& BuiltinFamilies() {
  static const std::array families = {families::Llama(), families::Qwen3(),
                                      families::Qwen3Moe(), families::Qwen3Next()};
  return families;
}

}  // namespace

StatusOr<const Family*> ResolveFamily(const CheckpointConfig& config) {
  const Family* by_type = nullptr;
  const Family* by_arch = nullptr;
  for (const auto& family : BuiltinFamilies()) {
    if (!config.model_type.empty() && config.model_type == family.model_type) by_type = &family;
    if (!config.architectures.empty() && config.architectures == family.architecture)
      by_arch = &family;
  }
  if (by_type && by_arch && by_type != by_arch) {
    return InvalidArgumentError("model_type and architectures select different model families");
  }
  if (by_type) return by_type;
  if (by_arch) return by_arch;
  return UnimplementedError("unsupported model architecture: ", config.model_type, " (",
                            config.architectures, ")");
}

StatusOr<std::unique_ptr<Model>> BuildFamily(const Family& family,
                                             models::LoadedCheckpoint& checkpoint,
                                             DeviceId device, int max_tokens, int max_seqs,
                                             const ParallelConfig& parallel) {
  if (family.build == nullptr) return UnimplementedError("model family has no builder");
  INFERX_ASSIGN_OR_RETURN(auto json, causal::ParseConfigJson(checkpoint.config_json));
  try {
    return family.build(json, checkpoint, device, max_tokens, max_seqs, parallel);
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError("invalid model configuration: ", e.what());
  }
}

StatusOr<causal::DecoderConfig> TranslateFamilyConfig(const CheckpointConfig& identity,
                                                      std::string_view config_json) {
  INFERX_ASSIGN_OR_RETURN(const Family* family, ResolveFamily(identity));
  if (family->translate == nullptr) {
    return UnimplementedError("model family has no decoder configuration translator");
  }
  INFERX_ASSIGN_OR_RETURN(auto json, causal::ParseConfigJson(config_json));
  try {
    INFERX_ASSIGN_OR_RETURN(auto config, family->translate(json));
    INFERX_RETURN_IF_ERROR(config.Validate());
    return config;
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError("invalid model configuration: ", e.what());
  }
}

StatusOr<std::unique_ptr<Model>> BuildModel(models::LoadedCheckpoint& checkpoint,
                                            DeviceId device, int max_tokens, int max_seqs,
                                            const ParallelConfig& parallel) {
  INFERX_ASSIGN_OR_RETURN(const Family* family, ResolveFamily(checkpoint.config));
  return BuildFamily(*family, checkpoint, device, max_tokens, max_seqs, parallel);
}

}  // namespace inferx
