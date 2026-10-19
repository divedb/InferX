#ifndef INFERX_ATTN_METADATA_H_
#define INFERX_ATTN_METADATA_H_

#include <array>
#include <cstdint>
#include <span>
#include <type_traits>

#include "inferx/cache/kv_block_pool.h"
#include "inferx/ops/attention/types.h"
#include "inferx/ops/op_context.h"

namespace inferx::attn {

/// The planner's group count is small; a fixed maximum keeps StepContext
/// a plain aggregate. The tree's CachePlan never approaches this today.
inline constexpr int kMaxCacheGroups = 8;

/// \brief How a step's batch is composed (docs/attention.md §6.3).
enum class BatchShape : std::uint8_t { kDecodeOnly, kPrefillOnly, kMixed };

/// \brief Host-side, backend-agnostic batch description. Built by the
///        runner from SchedulerOutput + CacheManager views; spans point
///        into step-lifetime storage (no copies).
struct BatchDesc {
  std::int32_t num_reqs = 0;
  std::int32_t num_tokens = 0;
  /// First num_decodes requests (decodes-first runner ordering).
  std::int32_t num_decodes = 0, num_decode_tokens = 0;
  std::int32_t num_prefills = 0, num_prefill_tokens = 0;
  std::int32_t max_query_len = 0, max_seq_len = 0;
  /// All requests share one query_len (graph-friendly).
  bool uniform_query_len = false;
  /// [num_reqs].
  std::span<const std::int32_t> query_lens;
  /// [num_reqs] Total incl. this step.
  std::span<const std::int32_t> seq_lens;
  /// [num_tokens] In-sequence positions.
  std::span<const std::int32_t> positions;
  /// Per cache group: flattened block tables, sequence after sequence (the
  /// existing kv_indices CSR shape). kNullBlockId entries mark rolled-off
  /// positions (window groups) — kernels skip null-block rows.
  std::array<std::span<const BlockId>, kMaxCacheGroups> block_ids{};
  /// Flattened block-table length per group.
  std::array<std::int32_t, kMaxCacheGroups> block_ids_num_pages{};
  /// Per cache group: recurrent-state slot per request (-1 if unused).
  std::array<std::span<const std::int32_t>, kMaxCacheGroups> state_slots{};

  [[nodiscard]] BatchShape Shape() const noexcept {
    return num_prefills == 0 ? BatchShape::kDecodeOnly
         : num_decodes == 0 ? BatchShape::kPrefillOnly
                            : BatchShape::kMixed;
  }
};

/// \brief Device-side, POD, per cache group. Pointers reference *persistent*
///        buffers (stable addresses); written only by the metadata builder.
///        This is models::AttentionMetadata generalized per group: the same
///        CSR arrays today's kernels consume, plus the group's window floor
///        and an opaque backend plan.
struct GroupMeta {
  // Per-sequence CSR (device):
  /// [B+1] query extents (this step's new tokens).
  const std::int32_t* qo_indptr = nullptr;
  /// [B+1] page-table extents of the full KV extent.
  const std::int32_t* kv_indptr = nullptr;
  /// [kv_pages] block ids, sequence after sequence.
  const std::int32_t* kv_indices = nullptr;
  /// [B] tokens in each sequence's last page.
  const std::int32_t* last_page_len = nullptr;
  // Per-token (device):
  /// [T] sequence slot per token.
  const std::int32_t* batch_indices = nullptr;
  /// [T] in-sequence position per token.
  const std::int32_t* positions = nullptr;
  // Per-group scalars:
  /// Tokens per page (plan.block_size).
  std::int32_t block_size = 0;
  /// Tokens below this group's retention floor are skipped by the kernel
  /// (from CacheManager::GroupSkippedTokens).
  std::int32_t window_left = 0;
  // Scalars for grid sizing / shape dispatch only — never the source of
  // truth for sizes under graph capture: kernels read device arrays.
  std::int32_t num_reqs = 0, num_tokens = 0;
  std::int32_t num_decodes = 0, num_decode_tokens = 0;
  std::int32_t num_prefills = 0, num_prefill_tokens = 0;
  std::int32_t max_query_len = 0, max_seq_len = 0;
  BatchShape shape = BatchShape::kDecodeOnly;
  /// Backend-private plan (e.g. split-KV schedule), device-resident,
  /// opaque to everyone else.
  const void* plan = nullptr;
};
static_assert(std::is_trivially_copyable_v<GroupMeta>);

/// \brief Passed to every layer of one step. Built once per step.
struct StepContext {
  ops::OpContext* ctx = nullptr;
  std::array<GroupMeta, kMaxCacheGroups> groups{};
  std::uint16_t num_groups = 0;
  WorkspaceRef workspace{};

  [[nodiscard]] const GroupMeta& Group(std::uint16_t g) const noexcept { return groups[g]; }
  [[nodiscard]] ops::OpContext& Ctx() const noexcept { return *ctx; }
  [[nodiscard]] const WorkspaceRef& Workspace() const noexcept { return workspace; }
};

}  // namespace inferx::attn

#endif  // INFERX_ATTN_METADATA_H_
