/// \file
/// \brief Checkpoint tensor-name mapping for the shared Llama-style layout.

#ifndef INFERX_MODELS_CAUSAL_WEIGHT_MAPPING_H_
#define INFERX_MODELS_CAUSAL_WEIGHT_MAPPING_H_

#include <string_view>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/engine/parallel_config.h"
#include "inferx/models/causal/decoder_stack.h"
#include "inferx/models/checkpoint.h"

namespace inferx::causal {

/// \brief Checkpoint names belong to the architecture family, not the
///        components. This is the Llama-style convention shared by llama,
///        qwen, mistral, phi, internlm, and friends.
struct CheckpointLayout {
  std::string backbone_prefix = "model.";
  std::string head_name = "lm_head.weight";
  std::string attention_name = "self_attn.";
  std::string feed_forward_name = "mlp.";
};

/// \brief Uploads one named checkpoint tensor to `device` as bfloat16,
///        checking it against `expected`. The mapping workhorse families
///        compose their models from.
StatusOr<Tensor> LoadWeight(const models::Checkpoint& checkpoint, std::string_view name,
                            const Shape& expected, DeviceId device);

/// \brief Loads one rank's vocab-dim row shard of `name`
///        (VocabParallelEmbedding analogue), shape-checked against the
///        full [vocab, hidden] tensor.
///
/// At tensor_parallel_size == 1 this is the whole embedding.
StatusOr<Tensor> LoadVocabShard(const models::Checkpoint& checkpoint, std::string_view name,
                                int64_t vocab, int64_t hidden, const ParallelConfig& parallel,
                                DeviceId device);

/// \brief Maps every decoder tensor (embedding, final norm, all layers) of
/// `checkpoint` through `layout`, with shape checks before upload.
///
/// `config` carries TOTAL head counts; `parallel` selects this rank's row
/// slices of the QKV projections.
///
/// Recurrent projection unpacking remains architecture-specific and
/// unsupported.
StatusOr<DecoderWeights> LoadDecoderWeights(const models::Checkpoint& checkpoint,
                                            const DecoderConfig& config,
                                            const CheckpointLayout& layout,
                                            const ParallelConfig& parallel, DeviceId device);

}  // namespace inferx::causal

#endif  // INFERX_MODELS_CAUSAL_WEIGHT_MAPPING_H_
