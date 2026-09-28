#pragma once

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Assembles per-head K/V for decompressed multi-latent attention.
///
/// K heads are [rope | nope] with the shared rotated slice first; V heads
/// copy their slice and zero the rope-sized tail so one head_dim serves
/// both caches and the padded output projection columns contribute
/// nothing. `k_rope` is [rows, rope_dim]; `up-projected` is
/// [rows, heads * (nope + v)]; outputs are [rows, heads, head_dim] with
/// head_dim = nope + rope.
Status AssembleMlaCaches(ExecutionContext& ctx, const Tensor& k_rope, const Tensor& up_projected,
                         int64_t heads, int64_t nope, int64_t rope, int64_t v_dim, Tensor& k_out,
                         Tensor& v_out);

}  // namespace inferx::ops
