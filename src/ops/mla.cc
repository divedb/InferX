#include "inferx/ops/mla.h"

#include "inferx/ops/cuda/mla.h"

namespace inferx::ops {

Status AssembleMlaCaches(OpContext& ctx, const Tensor& k_rope, const Tensor& up_projected,
                         int64_t heads, int64_t nope, int64_t rope, int64_t v_dim, Tensor& k_out,
                         Tensor& v_out) {
  const int64_t head_dim = nope + rope;
  if (k_rope.Rank() != 2 || up_projected.Rank() != 2 || k_out.Rank() != 3 || v_out.Rank() != 3 ||
      k_rope.GetDataType() != DataType::kBFloat16 ||
      up_projected.GetDataType() != DataType::kBFloat16 ||
      k_out.GetDataType() != DataType::kBFloat16 || v_out.GetDataType() != DataType::kBFloat16 ||
      k_rope.Dim(1) != rope || up_projected.Dim(1) != heads * (nope + v_dim) ||
      k_out.Dim(0) != k_rope.Dim(0) || k_out.Dim(1) != heads || k_out.Dim(2) != head_dim ||
      v_out.Dim(0) != k_rope.Dim(0) || v_out.Dim(1) != heads || v_out.Dim(2) != head_dim) {
    return InvalidArgumentError("MLA cache assembly shapes disagree");
  }
  for (const Tensor* t : std::initializer_list<const Tensor*>{
           &k_rope, &up_projected, &k_out, &v_out}) {
    if (t->Device() != ctx.Device()) {
      return InvalidArgumentError("MLA cache assembly must run on the context device");
    }
  }
  if (v_dim > head_dim || rope <= 0 || nope <= 0 || v_dim <= 0) {
    return InvalidArgumentError("MLA dimensions must be positive with v within head_dim");
  }
  if (!ctx.Device().IsCuda()) return UnimplementedError("MLA cache assembly requires CUDA");
  if (k_rope.IsEmpty()) return OkStatus();
  return cuda::AssembleMlaCaches(ctx, k_rope, up_projected, heads, nope, rope, v_dim, k_out, v_out);
}

}  // namespace inferx::ops
