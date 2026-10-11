#include "inferx/models/layers/linear.h"

#include <utility>

#include "inferx/ops/linear.h"

namespace inferx {
namespace models {

StatusOr<Linear> Linear::Create(int64_t in_features, int64_t out_features, DataType dtype,
                                Device device) {
  if (in_features <= 0 || out_features <= 0) {
    return InvalidArgumentError("Linear needs positive feature counts, got ",
                                in_features, " -> ", out_features);
  }
  INFERX_ASSIGN_OR_RETURN(Tensor weight, Tensor::Empty(dtype, Shape({out_features, in_features}),
                                                       device));
  return Linear(std::move(weight), in_features, out_features);
}

Status Linear::Forward(ops::OpContext& ctx, const Tensor& x, Tensor& out) const {
  return ops::Linear(ctx, x, weight_, out);
}

}  // namespace models
}  // namespace inferx
