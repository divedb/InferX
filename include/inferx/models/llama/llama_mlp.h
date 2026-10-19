#ifndef INFERX_MODELS_LLAMA_LLAMA_MLP_H_
#define INFERX_MODELS_LLAMA_LLAMA_MLP_H_

#include <cstdint>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/mlp.h"
#include "inferx/models/llama/llama_config.h"
#include "inferx/dist/shard.h"
#include "inferx/models/model.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace llama {

/// \brief Llama's SwiGLU feed-forward block (silu gating, like Qwen3's).
class LlamaMLP {
 public:
  LlamaMLP() = default;

  static StatusOr<LlamaMLP> Create(const LlamaConfig& config, const ParallelConfig& parallel,
                                   DataType dtype, Device device);

  Status Forward(ForwardContext& ctx, const Tensor& hidden, Tensor& out) const;

  Status BindWeights(const std::string& prefix, WeightMap& weights);

  /// \brief Declarative manifest: gate/up row-split (dst empty, fused
  ///        buffer), down column-split (dst engaged).
  [[nodiscard]] std::vector<ParamDecl> DescribeParams(const std::string& prefix) const;

 private:
  LlamaMLP(MLP mlp, int64_t tp_rank, int64_t hidden, int64_t global_intermediate)
      : mlp_(std::move(mlp)), tp_rank_(tp_rank), hidden_(hidden),
        global_intermediate_(global_intermediate) {}

  MLP mlp_;
  int64_t tp_rank_ = 0;
  int64_t hidden_ = 0;
  int64_t global_intermediate_ = 0;
};

}  // namespace llama
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_LLAMA_LLAMA_MLP_H_
