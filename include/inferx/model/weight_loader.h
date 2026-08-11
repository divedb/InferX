#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/types/span.h"
#include "inferx/core/device.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"
#include "inferx/model/safetensors.h"

namespace inferx::model {

/// \brief Where a load's time went, split at the seam that matters.
///
/// `stage_seconds` is host-side work: faulting mmap'd checkpoint pages in and
/// copying (or gathering) them into pinned staging. `h2d_wait_seconds` is time
/// spent blocked on the copy engine — waiting for a staging slot to drain or
/// for the final stream sync. A cold load should be stage-dominated (disk); a
/// warm load h2d-dominated (PCIe). Any other profile means the pipeline is
/// broken, and this split is what makes that observable.
struct WeightLoaderStats {
  size_t tensors = 0;
  /// Logical tensor bytes staged so far (not device bytes reserved).
  size_t bytes = 0;
  double stage_seconds = 0.0;
  double h2d_wait_seconds = 0.0;
};

/// \brief The one way weights move from a `Checkpoint` to the device.
///
/// `Checkpoint` is the lookup layer: name → borrowed mmap-backed host tensor.
/// This is the movement layer, shared by every architecture so the throughput
/// machinery exists once. Each verb lowers its source — a whole tensor, a
/// stack of tensors, a row permutation — to a list of contiguous host extents
/// and streams them through a fixed pinned ring: worker threads pack a slot
/// (parallelizing the page-fault + memcpy that dominate a cold load), a
/// asynchronous copies drain it at link rate while the next slot packs. Device
/// memory is bump-allocated from large chunks rather than one allocation
/// per tensor.
///
/// The contract that buys the overlap: **a returned view's bytes are not on
/// the device until `Finish()` (or `Release()`) has returned.** Load
/// everything, then finish, then run — the loader is for startup, not for
/// on-demand paging, which the serving path deliberately does not do.
///
/// With `Options::device` set to the CPU the same verbs gather into host
/// memory directly — no CUDA calls at all — so the transform logic is
/// testable without a device.
class WeightLoader {
 public:
  struct Options {
    DeviceId device;

    /// Pinned staging: `staging_slots` × `staging_slot_bytes`, the loader's
    /// entire pinned footprint. Two slots suffice for overlap; three keep the
    /// copy engine fed when host packing is jittery.
    size_t staging_slot_bytes = size_t{32} << 20;
    int staging_slots = 3;

    /// Host-side copy threads, the calling thread included. 0 picks a default
    /// from the hardware; staging is memory-bound, so a handful saturates.
    int threads = 0;

    /// Device memory comes from chunks of this size, bump-allocated. A tensor
    /// larger than a chunk gets a dedicated allocation of exactly its size.
    size_t device_chunk_bytes = size_t{256} << 20;
  };

  /// \brief Creates a loader over `checkpoint`, which must outlive it.
  static StatusOr<WeightLoader> Create(const Checkpoint* checkpoint,
                                       const Options& options);

  ~WeightLoader();

  WeightLoader(const WeightLoader&) = delete;
  WeightLoader& operator=(const WeightLoader&) = delete;
  WeightLoader(WeightLoader&&) noexcept;
  WeightLoader& operator=(WeightLoader&&) noexcept;

  /// \brief Uploads one tensor, shape-checked against the checkpoint.
  StatusOr<TensorView> Load(std::string_view name, const Shape& expected);

  /// \brief Uploads one tensor with whatever dtype and shape the checkpoint
  ///        records — for models whose shapes are validated downstream.
  StatusOr<TensorView> Load(std::string_view name);

  /// \brief Uploads several same-shaped tensors end to end as one tensor.
  ///
  /// Every source must match `part` (and share a dtype); the result is shaped
  /// `out`, whose byte size must equal the sum of the parts. This one verb is
  /// both fusion (gate|up → `[2·inter, h]`) and expert stacking (`E` or `2E`
  /// parts → `[E, …]`) — the difference is only the shape the caller declares.
  StatusOr<TensorView> LoadStacked(absl::Span<const std::string> names,
                                   const Shape& part, const Shape& out);

  /// \brief Uploads a 2-D tensor with destination row `i` reading source row
  ///        `row_map[i]` — the RoPE de-interleave case.
  StatusOr<TensorView> LoadRowPermuted(std::string_view name,
                                       const Shape& expected,
                                       absl::Span<const int64_t> row_map);

  /// \brief Uploads an already-materialized host tensor.
  ///
  /// The escape hatch for callers that transform between lookup and upload —
  /// TP sharding, small host-side gathers. Every verb stages its source
  /// completely before returning (only the pinned slots are read
  /// asynchronously), so the source may be a temporary.
  StatusOr<TensorView> Upload(const Tensor& host);

  /// \brief Uploads host tensors end to end as one `out`-shaped tensor.
  ///
  /// Unlike `LoadStacked`, parts need not share a shape — QKV fusion stacks
  /// `[q_dim, h]` with two `[kv_dim, h]` — only a dtype, and `out`'s byte
  /// size must equal the parts' sum.
  StatusOr<TensorView> UploadStacked(absl::Span<const Tensor> parts,
                                     const Shape& out);

  /// \brief Drains the pipeline. Only after this returns are all views valid.
  Status Finish();

  /// \brief Finishes, then hands over the device chunks backing every view.
  ///
  /// The caller keeps them alive for as long as the views are used. Terminal:
  /// the loader holds nothing afterwards and must not load more.
  StatusOr<std::vector<DeviceBuffer>> Release();

  /// \brief Running totals; consistent after every verb (no drain needed).
  const WeightLoaderStats& stats() const;

  /// \brief The resolved host-copy thread count.
  int threads() const;

  /// \brief Device buffers created so far.
  ///
  /// With `device_chunk_bytes = 0` every tensor gets a dedicated buffer, so
  /// `buffer_count() - 1` right after a verb is that tensor's index in the
  /// `Release()`d vector — for callers that must later free individual
  /// tensors (qwen2's FP8/INT4 conversion releases the bf16 originals).
  size_t buffer_count() const;

 private:
  struct Impl;

  explicit WeightLoader(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

/// \brief The shape of `parts` stacked along dim 0: rows sum, widths must
///        agree (1-D parts sum plainly).
///
/// The companion to `UploadStacked` for QKV-style fusions, computing the
/// declared shape from the (possibly TP-sharded) parts actually fetched
/// rather than from the config.
StatusOr<Shape> ConcatenatedShape(absl::Span<const Tensor> parts);

}  // namespace inferx::model
