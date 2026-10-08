/// \file
/// \brief The generic weight loader: maps checkpoint tensors into the
/// shared layers' canonical weights -- tensor access, transformation,
/// upload, and shape validation live here. Architectures declare naming
/// and fusion through WeightNames/WeightLayout; shared layers declare
/// parameter shapes through their configs.

#ifndef INFERX_MODELS_LOADING_WEIGHT_LOADER_H_
#define INFERX_MODELS_LOADING_WEIGHT_LOADER_H_

#include <string>
#include <string_view>
#include <vector>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/models/layers/decoder.h"
#include "inferx/models/loading/checkpoint_reader.h"
#include "inferx/models/loading/weight_mapping.h"

namespace inferx::models::loading {

/// \brief Loads this rank's vocabulary-dim row shard of `name`
///        (VocabParallelEmbedding analogue), shape-checked against the
///        full [vocab, hidden] tensor.
///
/// At tensor_parallel_size == 1 this is the whole embedding.
StatusOr<Tensor> LoadVocabShard(const Checkpoint& checkpoint, std::string_view name,
                                int64_t vocab, int64_t hidden, const ParallelConfig& parallel,
                                DeviceId device);

/// \brief Loads the fused, head-agnostic QKV projection (packed
///        [query (| gate) | key | value] rows), the row-parallel output
///        projection, and the optional biases, head norms, and sinks.
///
/// Allocates one packed device tensor and uploads this rank's row slices
/// of the checkpoint's q/k/v projections straight into it. The returned
/// per-projection weights are views into the packed allocation. `config`
/// carries TOTAL head counts; the shard geometry derives from them.
StatusOr<layers::AttentionWeights> LoadAttentionWeights(
    const Checkpoint& checkpoint, const std::string& prefix, const WeightNames& names,
    const WeightLayout& layout, const layers::AttentionConfig& config,
    const ParallelConfig& parallel, int64_t hidden, DeviceId device);

/// \brief Loads one gated-MLP block: fused gate|up rows plus the
///        row-parallel down projection.
StatusOr<layers::GatedMlpWeights> LoadGatedMlp(const Checkpoint& checkpoint,
                                               const std::string& gate_name,
                                               const std::string& up_name,
                                               const std::string& down_name, int64_t hidden,
                                               int64_t total_intermediate,
                                               const ParallelConfig& parallel, DeviceId device);

/// \brief Loads MLA projections with the layout the kernels assume: q_b
///        rows reorder per head [nope | rope] -> [rope | nope], kv_a splits
///        into its rope slice and latent, and o_proj gains zero columns
///        for the padded V tail.
StatusOr<layers::MlaWeights> LoadMlaWeights(const Checkpoint& checkpoint,
                                            const std::string& prefix, const WeightNames& names,
                                            const layers::MlaConfig& config, int64_t hidden,
                                            DeviceId device);

/// \brief Loads the Gated DeltaNet projections and float32 gates.
StatusOr<layers::GdnWeights> LoadGdnWeights(const Checkpoint& checkpoint,
                                            const std::string& prefix, const WeightNames& names,
                                            const layers::GatedDeltaNetConfig& config,
                                            int64_t hidden, DeviceId device);

/// \brief Loads every decoder tensor (embedding, final norm, all layers)
///        of `checkpoint` through `names` and `layout`, with shape checks
///        before upload. `config` carries TOTAL head counts; `parallel`
///        selects this rank's slices.
StatusOr<layers::StackWeights> LoadStackWeights(const Checkpoint& checkpoint,
                                                const layers::StackConfig& config,
                                                const WeightNames& names,
                                                const WeightLayout& layout,
                                                const ParallelConfig& parallel, DeviceId device);

}  // namespace inferx::models::loading

#endif  // INFERX_MODELS_LOADING_WEIGHT_LOADER_H_
