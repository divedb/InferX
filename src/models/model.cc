#include "inferx/models/model.h"

#include "inferx/models/llama/llama.h"
#include "inferx/models/qwen3/qwen3.h"

namespace inferx {

StatusOr<std::unique_ptr<Model>> Model::Load(const std::string& directory, DeviceId device,
                                             int max_tokens, int max_seqs) {
  INFERX_ASSIGN_OR_RETURN(ModelConfig config,
                          ModelConfig::FromFile(directory + "/config.json"));
  // Architecture directories own their implementations; new architectures
  // register here and nowhere else.
  if (config.model_type == "qwen3") {
    return Qwen3Model::Load(directory, device, max_tokens, max_seqs);
  }
  if (config.model_type == "llama") {
    return LlamaModel::Load(directory, device, max_tokens, max_seqs);
  }
  return UnimplementedError("unsupported model architecture: ", config.model_type, " (",
                            config.architectures, ")");
}

}  // namespace inferx
