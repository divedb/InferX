#ifndef INFERX_MODELS_QWEN3_QWEN3_MLP_H_
#define INFERX_MODELS_QWEN3_QWEN3_MLP_H_

#include <cstdint>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/layers/mlp.h"
#include "inferx/dist/shard.h"
#include "inferx/models/model.h"
#include "inferx/models/qwen3/qwen3_config.h"
#include "inferx/models/weight_loader.h"

namespace inferx {
namespace models {
namespace qwen3 {

/// \brief Qwen3's SwiGLU feed-forward block.
///
/// A thin, explicit wrapper: Qwen3's MLP is exactly the reusable gated
/// projection with no family quirks, and keeping the class makes the
/// decoder layer read as the architecture.
class Qwen3MLP {
 public:
  Qwen3MLP() = default;

  static StatusOr<Qwen3MLP> Create(const Qwen3Config& config, const ParallelConfig& parallel,
                                   DataType dtype, Device device);

  Status Forward(ForwardContext& ctx, const Tensor& hidden, Tensor& out) const;

  /// \brief Binds `prefix.{gate,up,down}_proj.weight`.
  Status BindWeights(const std::string& prefix, WeightMap& weights);

  /// \brief Declarative manifest: gate/up rows split over TP into the
  ///        packed [gate | up] buffer (dst empty, model-side placement),
  ///        down column-split (dst engaged).
  [[nodiscard]] std::vector<ParamDecl> DescribeParams(const std::string& prefix) const;


  int64_t IntermediateShard() const { return mlp_.IntermediateShard(); }

 private:
  Qwen3MLP(MLP mlp, int64_t tp_rank, int64_t hidden, int64_t global_intermediate)
      : mlp_(std::move(mlp)), tp_rank_(tp_rank), hidden_(hidden),
        global_intermediate_(global_intermediate) {}

  MLP mlp_;
  int64_t tp_rank_ = 0;
  int64_t hidden_ = 0;
  int64_t global_intermediate_ = 0;
};

}  // namespace qwen3
}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_QWEN3_QWEN3_MLP_H_
