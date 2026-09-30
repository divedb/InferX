#pragma once

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/op_context.h"

namespace inferx::ops {

/// \brief Gathers rows of `src` into `out`: out[i, :] <- src[indices[i], :].
///
/// Dtype-agnostic byte copy, so it serves both embedding lookups (indices are
/// token ids) and activation row selection (indices are output rows). Rows
/// are independent; `out` must not alias `src`.
///
/// \param ctx      Execution context; all tensors must live on ctx.Device().
/// \param src      [rows, dim] source rows.
/// \param indices  [count] int32 row indices into `src`.
/// \param out      [count, dim] gathered rows, same dtype as `src`.
/// \return         OK, or InvalidArgument/Unimplemented for bad inputs.
Status GatherRows(OpContext& ctx, const Tensor& src, const Tensor& indices,
                  Tensor& out);

/// \brief Gathers rows of a shard of `src`, zeroing out-of-shard indices:
///        out[i, :] <- (0 <= indices[i] - row_begin < src rows) ?
///                     src[indices[i] - row_begin, :] : 0.
///
/// The vocab-parallel embedding lookup: every rank holds one row range of
/// the table, so token ids outside the range gather as zero rows and the
/// following all-reduce completes the embedding.
Status GatherRowsRange(OpContext& ctx, const Tensor& src, const Tensor& indices,
                       Tensor& out, int64_t row_begin);

/// \brief Copies a rank-2 block into a column range of a wider rank-2
///        tensor: dst[i, col_begin + j] <- src[i, j].
///
/// The assembly step of an all-gather along the last dimension: each rank
/// writes its shard into its column block of the shared result.
Status CopyColumnBlock(OpContext& ctx, const Tensor& src, Tensor& dst,
                       int64_t col_begin);

}  // namespace inferx::ops
