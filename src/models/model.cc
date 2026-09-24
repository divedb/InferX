#include "inferx/models/model.h"

#include "inferx/core/logging.h"
#include "inferx/models/checkpoint.h"
#include "inferx/models/model_registry.h"

namespace inferx {

StatusOr<std::unique_ptr<Model>> Model::Load(const std::string& directory, DeviceId device,
                                             int max_tokens, int max_seqs,
                                             const ParallelConfig& parallel) {
  INFERX_LOG(INFO) << "loading model from " << directory;
  // Identity first: an unsupported architecture is rejected before any
  // weight shard is mapped.
  INFERX_ASSIGN_OR_RETURN(auto checkpoint, models::LoadCheckpointConfig(directory));
  INFERX_ASSIGN_OR_RETURN(const Family* family, ResolveFamily(checkpoint.config));
  INFERX_LOG(INFO) << "model family " << family->model_type << " ("
                   << checkpoint.config.architectures << ")";
  INFERX_RETURN_IF_ERROR(models::OpenCheckpointWeights(checkpoint, directory));
  return BuildFamily(*family, checkpoint, device, max_tokens, max_seqs, parallel);
}

}  // namespace inferx
