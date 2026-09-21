#ifndef INFERX_MODELS_QWEN3_QWEN3_H_
#define INFERX_MODELS_QWEN3_QWEN3_H_

#include <memory>
#include <string>
#include <vector>

#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/model.h"
#include "inferx/ops/execution_context.h"

namespace inferx {

/// \brief Projection weights for one Qwen3 attention layer.
///
/// All matrices are stored exactly as the checkpoint stores them:
/// `[out_features, in_features]`, bfloat16, on the model's device.
struct Qwen3AttentionWeights {
  Tensor q_proj;  ///< [num_heads * head_dim, hidden]
  Tensor k_proj;  ///< [num_kv_heads * head_dim, hidden]
  Tensor v_proj;  ///< [num_kv_heads * head_dim, hidden]
  Tensor o_proj;  ///< [hidden, num_heads * head_dim]
  Tensor q_norm;  ///< [head_dim] per-head query RMSNorm.
  Tensor k_norm;  ///< [head_dim] per-head key RMSNorm.
};

/// \brief Feed-forward projection weights for one layer.
struct Qwen3MlpWeights {
  Tensor gate_proj;  ///< [intermediate, hidden]
  Tensor up_proj;    ///< [intermediate, hidden]
  Tensor down_proj;  ///< [hidden, intermediate]
};

/// \brief All weights belonging to a single transformer layer.
struct Qwen3LayerWeights {
  Tensor input_layernorm;           ///< [hidden]
  Tensor post_attention_layernorm;  ///< [hidden]
  Qwen3AttentionWeights attn;
  Qwen3MlpWeights mlp;
};

/// \brief Qwen3 dense decoder-only checkpoint (e.g. Qwen3-0.6B) in bfloat16.
struct Qwen3Weights {
  ModelConfig config;   ///< Parsed configuration.
  Tensor embed_tokens;  ///< [vocab, hidden]
  Tensor lm_head;       ///< [vocab, hidden]
  Tensor norm;          ///< Final RMSNorm weight, [hidden].
  std::vector<Qwen3LayerWeights> layers;

  /// \brief Loads `config.json` + `model.safetensors` from `model_dir`.
  static StatusOr<Qwen3Weights> Load(const std::string& model_dir, DeviceId device);
};

/// \brief Qwen3 architecture: weights plus its own forward flow.
class Qwen3Model final : public Model {
 public:
  /// \brief Loads the checkpoint and prepares the execution workspace.
  static StatusOr<std::unique_ptr<Model>> Load(const std::string& directory, DeviceId device,
                                               int max_tokens, int max_seqs);

  const ModelConfig& config() const override { return weights_.config; }
  StatusOr<Tensor> Forward(const ModelInput& input, KvBlockPool& cache,
                           ops::ExecutionContext& ctx) override;

 private:
  explicit Qwen3Model(Qwen3Weights&& weights);

  Qwen3Weights weights_;
};

}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_QWEN3_H_
