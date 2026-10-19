#ifndef INFERX_MODELS_QWEN3_QWEN3_CAUSAL_LM_H_
#define INFERX_MODELS_QWEN3_QWEN3_CAUSAL_LM_H_

#include <memory>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/embedding.h"
#include "inferx/models/layers/linear.h"
#include "inferx/models/load_options.h"
#include "inferx/models/model.h"
#include "inferx/models/model_spec.h"
#include "inferx/models/qwen3/qwen3_config.h"
#include "inferx/models/qwen3/qwen3_model.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace qwen3 {

/// \brief Qwen3ForCausalLM: vocab-parallel embedding, the Qwen3 tower, and
///        a column-parallel LM head (tied to the embedding when the
///        checkpoint says so).
///
/// This is the class the registry hands the runner; it is the only Qwen3
/// class that implements `Model`.
class Qwen3ForCausalLM final : public Model, public WeightBindable {
 public:
  /// \brief Builds the model for this rank from a parsed checkpoint config.
  static StatusOr<std::unique_ptr<Qwen3ForCausalLM>> Create(const Qwen3Config& config,
                                                            const ParallelConfig& parallel,
                                                            DataType dtype,
                                                            Device device);

  const ModelSpec& GetModelSpec() const override { return spec_; }
  std::vector<attn::AttentionSpec> GetAttentionSpecs() const override {
    return model_.AttentionSpecs();
  }
  Status LoadWeights(CheckpointReader& checkpoint) override;
  Status Forward(ForwardContext& ctx, const StepInputs& inputs, Tensor& hidden) const override;
  Status ComputeLogits(ForwardContext& ctx, const Tensor& hidden, const Tensor& rows,
                       Tensor& logits) const override;
  Status BuildWeightMap(WeightMap& weights) override;

 private:
  Qwen3ForCausalLM(VocabParallelEmbedding embed_tokens, Qwen3Model model,
                   Linear lm_head, bool tied, int64_t tp_size, int64_t tp_rank,
                   ModelSpec spec)
      : embed_tokens_(std::move(embed_tokens)),
        model_(std::move(model)),
        lm_head_(std::move(lm_head)),
        tied_(tied),
        tp_size_(tp_size),
        tp_rank_(tp_rank),
        spec_(std::move(spec)) {}

  VocabParallelEmbedding embed_tokens_;
  Qwen3Model model_;
  Linear lm_head_;
  bool tied_ = false;
  int64_t tp_size_ = 1;
  int64_t tp_rank_ = 0;
  ModelSpec spec_;
};

}  // namespace qwen3
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_QWEN3_CAUSAL_LM_H_
