#ifndef INFERX_LAYERS_LINEAR_H_
#define INFERX_LAYERS_LINEAR_H_

#include <cstdint>

#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/op_context.h"

namespace inferx {
namespace models {

/// \brief Bias-free projection owning its weight, stored exactly as
///        checkpoints store it: [out_features, in_features].
///
/// The module is deliberately thin -- shape memory plus weight ownership
/// around `ops::Linear`. How a checkpoint's rows and columns reach this
/// rank (plain, column-parallel, row-parallel, fused into a bigger buffer)
/// is a weight-binding decision that stays in the model family: this class
/// computes whatever its held weight says. Models that need a collective
/// after the GEMM (row-parallel use) issue it themselves.
class Linear {
 public:
  Linear() = default;

  /// \brief Allocates an uninitialized weight on `device`.
  ///
  /// \param in_features  Input width; x columns.
  /// \param out_features Output width; weight rows.
  /// \param dtype        Activation dtype; bfloat16 today.
  /// \param device       Destination device.
  /// \return             OK, or the allocation status.
  static StatusOr<Linear> Create(int64_t in_features, int64_t out_features,
                                 DataType dtype, Device device);

  /// \brief Projects `x` [tokens, in] into `out` [tokens, out].
  ///
  /// \param ctx  Execution context; tensors live on its device.
  /// \param x    [tokens, in_features] input.
  /// \param out  [tokens, out_features] output; must not alias `x`.
  /// \return     OK, or InvalidArgument/Unimplemented from the GEMM.
  Status Forward(ops::OpContext& ctx, const Tensor& x, Tensor& out) const;

  const Tensor& Weight() const { return weight_; }
  Tensor* MutableWeight() { return &weight_; }
  int64_t InFeatures() const { return in_features_; }
  int64_t OutFeatures() const { return out_features_; }

 private:
  Linear(Tensor weight, int64_t in_features, int64_t out_features)
      : weight_(std::move(weight)),
        in_features_(in_features),
        out_features_(out_features) {}

  Tensor weight_;  ///< [out_features, in_features]
  int64_t in_features_ = 0;
  int64_t out_features_ = 0;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_LAYERS_LINEAR_H_
