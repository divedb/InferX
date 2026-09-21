#ifndef INFERX_MODELS_LLAMA_LLAMA_H_
#define INFERX_MODELS_LLAMA_LLAMA_H_

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

/// \brief Projection weights for one Llama attention layer.
///
/// All matrices are stored exactly as the checkpoint stores them:
/// `[out_features, in_features]`, bfloat16, on the model's device. Llama
/// applies no per-head Q/K normalization, unlike Qwen3.
struct LlamaAttentionWeights {
  Tensor q_proj;  ///< [num_heads * head_dim, hidden]
  Tensor k_proj;  ///< [num_kv_heads * head_dim, hidden]
  Tensor v_proj;  ///< [num_kv_heads * head_dim, hidden]
  Tensor o_proj;  ///< [hidden, num_heads * head_dim]
};

/// \brief Feed-forward projection weights for one layer (SwiGLU MLP).
struct LlamaMlpWeights {
  Tensor gate_proj;  ///< [intermediate, hidden]
  Tensor up_proj;    ///< [intermediate, hidden]
  Tensor down_proj;  ///< [hidden, intermediate]
};

/// \brief All weights belonging to a single transformer layer.
struct LlamaLayerWeights {
  Tensor input_layernorm;           ///< [hidden]
  Tensor post_attention_layernorm;  ///< [hidden]
  LlamaAttentionWeights attn;
  LlamaMlpWeights mlp;
};

/// \brief Llama dense decoder-only checkpoint (Llama 2/3 family) in bfloat16.
struct LlamaWeights {
  ModelConfig config;   ///< Parsed configuration.
  Tensor embed_tokens;  ///< [vocab, hidden]
  Tensor lm_head;       ///< [vocab, hidden]
  Tensor norm;          ///< Final RMSNorm weight, [hidden].
  std::vector<LlamaLayerWeights> layers;

  /// \brief Loads `config.json` + `model.safetensors` from `model_dir`.
  static StatusOr<LlamaWeights> Load(const std::string& model_dir, DeviceId device);
};

/// \brief Llama architecture: weights plus its own forward flow.
class LlamaModel final : public Model {
 public:
  /// \brief Loads the checkpoint and prepares the execution workspace.
  static StatusOr<std::unique_ptr<Model>> Load(const std::string& directory, DeviceId device,
                                               int max_tokens, int max_seqs);

  const ModelConfig& config() const override { return weights_.config; }
  StatusOr<Tensor> Forward(const ModelInput& input, KvBlockPool& cache,
                           ops::ExecutionContext& ctx) override;

 private:
  explicit LlamaModel(LlamaWeights&& weights);

  LlamaWeights weights_;
};

}  // namespace inferx

#endif  // INFERX_MODELS_LLAMA_LLAMA_H_
