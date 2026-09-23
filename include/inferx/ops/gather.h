#pragma once

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {

/// \brief Gathers rows of `src` into `out`: out[i, :] <- src[indices[i], :].
///
/// Dtype-agnostic byte copy, so it serves both embedding lookups (indices are
/// token ids) and activation row selection (indices are output rows). Rows
/// are independent; `out` must not alias `src`.
///
/// \param ctx      Execution context; all tensors must live on ctx.device().
/// \param src      [rows, dim] source rows.
/// \param indices  [count] int32 row indices into `src`.
/// \param out      [count, dim] gathered rows, same dtype as `src`.
/// \return         OK, or InvalidArgument/Unimplemented for bad inputs.
Status GatherRows(ExecutionContext& ctx, const Tensor& src, const Tensor& indices,
                  Tensor& out);

}  // namespace inferx::ops
