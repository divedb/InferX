#ifndef INFERX_ATTN_BUILDER_H_
#define INFERX_ATTN_BUILDER_H_

#include <cstdint>
#include <vector>

#include "inferx/core/device_buffer.h"
#include "inferx/core/status.h"
#include "inferx/ops/attention/metadata.h"
#include "inferx/ops/op_context.h"

namespace inferx::attn {

/// \brief A graph capture size: uniform decode-only batches only
///        (docs/attention.md §10).
struct CaptureShape {
  std::int32_t num_reqs = 0;
  std::int32_t query_len = 1;
  [[nodiscard]] std::int32_t NumTokens() const noexcept { return num_reqs * query_len; }
};

/// \brief Persistent device buffers backing one cache group's GroupMeta.
///
/// Sized once at init from EngineLimits. The shared per-token arrays
/// (qo_indptr, batch_indices, positions) are batch-wide, so the runtime
/// owns them once and every group's GroupMeta points at the same memory;
/// the page-table arrays (kv_indptr, kv_indices, last_page_len) differ per
/// group and live here.
struct DeviceMetaBuffers {
  DeviceBuffer qo_indptr;      ///< [max_reqs + 1]
  DeviceBuffer batch_indices;  ///< [max_tokens]
  DeviceBuffer positions;      ///< [max_tokens]
  DeviceBuffer kv_indptr;      ///< [max_reqs + 1]
  DeviceBuffer kv_indices;     ///< [max_reqs * max_blocks_per_req]
  DeviceBuffer last_page_len;  ///< [max_reqs]
};

/// \brief What Build() writes into: the persistent buffers plus the
///        execution context for the H2D copies.
struct BuildContext {
  ops::OpContext* ctx = nullptr;
  DeviceMetaBuffers* buffers = nullptr;
};

/// \brief Turns a BatchDesc into a device-resident GroupMeta once per step
///        per cache group. The only writer of GroupMeta; kernels read only.
class MetadataBuilder {
 public:
  virtual ~MetadataBuilder() = default;

  [[nodiscard]] virtual GraphSupport GetGraphSupport() const noexcept = 0;
  /// \brief Largest query_len the backend treats as "decode-like"
  ///        (1 = plain decode).
  [[nodiscard]] virtual std::int32_t DecodeThreshold() const noexcept = 0;

  /// \brief Hot path: O(num_reqs + num_tokens + pages). Stages the CSR
  ///        arrays and enqueues one async H2D per array group.
  [[nodiscard]] virtual Status Build(const BatchDesc& desc, std::uint16_t group,
                                     BuildContext& ctx, GroupMeta& out) noexcept = 0;

  /// \brief Builds metadata for a synthetic padded batch of exactly
  ///        `shape`, for graph capture/replay (docs/attention.md §10).
  [[nodiscard]] virtual Status BuildForCapture(const CaptureShape& shape, std::uint16_t group,
                                               BuildContext& ctx, GroupMeta& out) noexcept = 0;
};

/// \brief The common paged-KV metadata fill shared by all backends:
///        composition, not inheritance (docs/attention.md §6.4). Backend
///        builders call this first and append only their own `plan`.
[[nodiscard]] Status FillCommonMetadata(const BatchDesc& desc, std::uint16_t group,
                                        BuildContext& ctx, GroupMeta& out) noexcept;

/// rief The shared paged-KV builder backends return from CreateBuilder
///        unless they need their own plan (docs/attention.md §6.4).
[[nodiscard]] std::unique_ptr<MetadataBuilder> MakeCommonBuilder(GraphSupport graph);

}  // namespace inferx::attn

#endif  // INFERX_ATTN_BUILDER_H_
