#include "inferx/models/model.h"

#include "inferx/models/llama/builder.h"
#include "inferx/models/qwen3/builder.h"

namespace inferx {
namespace {

/// One row of the architecture registry: the checkpoint identity a family
/// claims, and the loader that builds it.
struct Family {
  const char* model_type;    ///< HF `model_type`, e.g. "llama".
  const char* architecture;  ///< HF `architectures[0]`, e.g. "LlamaForCausalLM".
  StatusOr<std::unique_ptr<Model>> (*load)(const std::string&, DeviceId, int, int, ops::AttentionBackend);
};

}  // namespace

StatusOr<std::unique_ptr<Model>> Model::Load(const std::string& directory, DeviceId device,
                                             int max_tokens, int max_seqs, ops::AttentionBackend backend) {
  INFERX_ASSIGN_OR_RETURN(CheckpointConfig config,
                          CheckpointConfig::FromFile(directory + "/config.json"));
  // Architecture directories own their implementations; new architectures
  // register in this table and nowhere else.
  static constexpr Family kFamilies[] = {
      {"llama", "LlamaForCausalLM", llama::Load},
      {"qwen3", "Qwen3ForCausalLM", qwen3::Load},
      {"qwen3_moe", "Qwen3MoeForCausalLM", qwen3::Load},
      {"qwen3_next", "Qwen3NextForCausalLM", qwen3::Load},
  };
  for (const auto& family : kFamilies) {
    if (config.model_type == family.model_type ||
        (!config.architectures.empty() && config.architectures == family.architecture)) {
      return family.load(directory, device, max_tokens, max_seqs, backend);
    }
  }
  return UnimplementedError("unsupported model architecture: ", config.model_type, " (",
                            config.architectures, ")");
}

}  // namespace inferx
