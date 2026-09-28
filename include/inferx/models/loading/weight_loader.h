/// \file
/// \brief Maps checkpoint tensors into canonical decoder weights.

#ifndef INFERX_MODELS_LOADING_WEIGHT_LOADER_H_
#define INFERX_MODELS_LOADING_WEIGHT_LOADER_H_

#include <string_view>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/models/causal/decoder_config.h"
#include "inferx/models/checkpoint.h"
#include "inferx/models/loading/weight_names.h"

namespace inferx::causal {

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
/// `checkpoint` through `names` and `layout`, with shape checks before upload.
///
/// `config` carries TOTAL head counts; `parallel` selects this rank's row
/// slices of the QKV projections.
///
/// Recurrent projection unpacking remains architecture-specific and
/// unsupported.
StatusOr<DecoderWeights> LoadDecoderWeights(const models::Checkpoint& checkpoint,
                                            const DecoderConfig& config,
                                            const models::WeightNames& names,
                                            const models::WeightLayout& layout,
                                            const ParallelConfig& parallel, DeviceId device);

}  // namespace inferx::causal

#endif  // INFERX_MODELS_LOADING_WEIGHT_LOADER_H_
