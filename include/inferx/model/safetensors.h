#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "inferx/core/dtype.h"
#include "inferx/core/shape.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"

namespace inferx::model {

/// \brief One tensor's entry in a safetensors header.
struct TensorEntry {
  DataType dtype = DataType::kUndefined;
  Shape shape;
  /// Byte range within the shard's data section, [begin, end).
  size_t begin = 0;
  size_t end = 0;
  /// Index into `Checkpoint::shard_paths()`.
  int shard = 0;

  size_t nbytes() const { return end - begin; }
};

/// \brief A read-only, mmap-backed safetensors checkpoint, possibly sharded.
///
/// The file is mapped, never read into a buffer, and every tensor handed out is
/// a **borrowed** `Storage` pointing into that mapping. Loading a 3B checkpoint
/// is therefore one `mmap` and 434 handles, not 434 allocations and 6 GB of
/// copying — which is the case T17 exists for. Nothing is on the device yet;
/// these are host tensors, and uploading is a separate, explicit step.
///
/// The mapping outlives every tensor derived from it because the `Checkpoint`
/// holds it and the tensors borrow from it. Destroying the `Checkpoint` while
/// tensors are still alive leaves them dangling — the one rule this class asks
/// callers to keep, and the reason it is move-only rather than copyable.
class Checkpoint {
 public:
  /// \brief Opens a checkpoint directory.
  ///
  /// Accepts either layout HuggingFace produces: a single `model.safetensors`,
  /// or an index (`model.safetensors.index.json`) naming several shards. Both
  /// end up as one flat name→tensor map, because nothing above this layer
  /// should care how the publisher chose to split the file.
  ///
  /// \param dir Directory containing the checkpoint.
  /// \return    The checkpoint, or NotFound/InvalidArgument describing what
  ///            about the directory did not work.
  static StatusOr<Checkpoint> Open(std::string_view dir);

  /// \brief Opens a single `.safetensors` file.
  static StatusOr<Checkpoint> OpenFile(std::string_view path);

  ~Checkpoint();

  Checkpoint(const Checkpoint&) = delete;
  Checkpoint& operator=(const Checkpoint&) = delete;
  Checkpoint(Checkpoint&&) noexcept;
  Checkpoint& operator=(Checkpoint&&) noexcept;

  /// \brief Looks up a tensor by its checkpoint name.
  ///
  /// The returned `Tensor` borrows the mapping: no copy, no allocation, and the
  /// bytes are paged in lazily by the kernel as they are touched.
  ///
  /// \param name Tensor name, e.g. `model.layers.0.self_attn.q_proj.weight`.
  /// \return     A host tensor over the mapping, or NotFound.
  StatusOr<Tensor> Get(std::string_view name) const;

  /// \brief Looks up a tensor and checks its shape, naming both in any error.
  ///
  /// Loading is where a shape mismatch is cheap to diagnose and expensive to
  /// miss: an unchecked wrong shape becomes a silently wrong GEMM much later.
  StatusOr<Tensor> GetChecked(std::string_view name, const Shape& expected) const;

  bool Contains(std::string_view name) const;

  const TensorEntry* FindEntry(std::string_view name) const;

  /// \brief Every tensor name, sorted, for diagnostics and conformance tests.
  std::vector<std::string> Names() const;

  size_t size() const;

  /// \brief Total bytes of tensor data across all shards.
  size_t TotalBytes() const;

  const std::vector<std::string>& shard_paths() const;

 private:
  struct Impl;

  explicit Checkpoint(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

/// \brief Maps a safetensors dtype string to a `DataType`.
///
/// Exposed because the loader is not the only thing that needs to agree with
/// the format's spelling — conformance tests read headers directly.
///
/// \param name The dtype as spelled in the header, e.g. "BF16".
/// \return     The corresponding DataType, or InvalidArgument.
StatusOr<DataType> SafetensorsDataType(std::string_view name);

}  // namespace inferx::model
