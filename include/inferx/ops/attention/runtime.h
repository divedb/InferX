#ifndef INFERX_ATTN_RUNTIME_H_
#define INFERX_ATTN_RUNTIME_H_

#include <memory>
#include <span>

#include "inferx/cache/cache_manager.h"
#include "inferx/core/status.h"
#include "inferx/ops/attention/backend.h"
#include "inferx/ops/attention/builder.h"
#include "inferx/ops/attention/metadata.h"
#include "inferx/ops/attention/op.h"
#include "inferx/ops/attention/registry.h"
#include "inferx/ops/attention/spec.h"
#include "inferx/ops/op_context.h"

namespace inferx::attn {

/// \brief Engine-facing knobs for one AttentionRuntime (docs/attention.md
///        §8.5). The runtime owns the whole stack it creates: plan, pools,
///        cache facade, kernels, builders, metadata buffers.
struct AttentionConfig {
  /// Cache plan block size; also the pools' page token count.
  std::int64_t block_size = 16;
  /// Planner's context length (window promotion bound).
  std::int64_t max_model_len = 8192;
  /// Concurrent sequences (state pool slots, metadata row counts).
  std::int64_t max_seqs = 256;
  /// Upper bound on one step's total query tokens (metadata buffers).
  std::int64_t max_batch_tokens = 8192;
  /// Total bytes the cache pools may allocate (state pool first, remainder
  /// to the block pool).
  std::int64_t cache_bytes_budget = 1LL << 30;
  /// Where kernels execute; selects CPU vs CUDA backends.
  Device device = Device::Cpu();
  bool enable_prefix_caching = true;
  /// "" = auto-select by capability and priority.
  std::string_view forced_backend;
};

/// \brief Composition root: turns attention specs into a running stack
///        (cache plan -> pools -> facade -> backend selection -> kernels ->
///        builders -> ops), then drives steps.
///
/// The cache facade is the scheduler-thread surface (AdoptPrefix / Reserve
/// / PublishComputed / Fork); BeginStep is the runner-thread surface.
class AttentionRuntime {
 public:
  /// \brief Builds everything; fails with the selection rejection trail if
  ///        no backend serves a spec, or with the planner's/pools' errors
  ///        for invalid configurations.
  [[nodiscard]] static StatusOr<std::unique_ptr<AttentionRuntime>> Create(
      std::unique_ptr<BackendRegistry> registry, const AttentionConfig& config,
      std::span<const AttentionSpec> specs);

  /// \brief The cache facade driving request lifecycle (scheduler thread).
  [[nodiscard]] inferx::CacheManager& Cache() const { return *cache_; }
  [[nodiscard]] const CachePlan& GetPlan() const { return plan_; }
  /// \brief Read-only block pool access for diagnostics and cache dumps.
  [[nodiscard]] const KvBlockPool* BlockPool() const { return pool_.get(); }

  /// \brief Group count (token-wise groups first, recurrent last).
  [[nodiscard]] std::uint16_t NumGroups() const { return num_groups_; }

  /// \brief Binds one attention layer (index into the `specs` span passed
  ///        to Create) to its kernel, cache view, and group.
  /// \brief Hands out the layer op; each layer hands out once.
  [[nodiscard]] StatusOr<AttentionOp> MakeOp(std::int64_t layer);

  /// \brief Builds per-group GroupMeta for one step: stages the CSR arrays
  ///        and enqueues the H2D copies. The returned StepContext (and the
  ///        BatchDesc it points at) must outlive the step's kernels.
  [[nodiscard]] StatusOr<StepContext> BeginStep(const BatchDesc& desc,
                                                ops::OpContext& ctx);

 private:
  AttentionRuntime() = default;

  std::unique_ptr<BackendRegistry> registry_;
  AttentionConfig config_{};
  std::unique_ptr<KvBlockPool> pool_;
  std::unique_ptr<RecurrentStatePool> state_pool_;
  std::unique_ptr<inferx::CacheManager> cache_;
  CachePlan plan_;
  std::unique_ptr<BackendSelector> selector_;

  /// Per layer (aligned with the specs span): bound kernel + view + group.
  struct BoundLayer {
    AttentionSpec spec;
    std::unique_ptr<Kernel> kernel;
    CacheView view;
    std::uint16_t group = 0;
  };
  std::vector<BoundLayer> layers_;

  std::vector<std::unique_ptr<MetadataBuilder>> builders_;
  std::vector<DeviceMetaBuffers> group_buffers_;
  std::uint16_t num_groups_ = 0;
};

}  // namespace inferx::attn

#endif  // INFERX_ATTN_RUNTIME_H_
