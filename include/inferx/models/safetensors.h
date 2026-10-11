#ifndef INFERX_MODELS_SAFETENSORS_H_
#define INFERX_MODELS_SAFETENSORS_H_

#include <string>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/models/loading/safe_tensors_reader.h"

namespace inferx {
namespace models {

/// \brief The tensors of one checkpoint directory, however many files hold
///        them, as zero-copy views over the restored SafeTensorReader
///        mappings.
///
/// Wraps `model.safetensors` when present, otherwise every `*.safetensors`
/// file in the directory (auxiliary checkpoints -- optimizers, adapters --
/// are skipped). The format layer (header parsing, offset validation, the
/// safetensors error taxonomy) lives entirely in
/// `loading/safe_tensors_{view,reader}`; this class only routes names to
/// shards. Tensors borrow the mappings, so they live exactly as long as
/// this reader.
class CheckpointReader {
 public:
  CheckpointReader() = default;

  /// \brief Maps the checkpoint stored in `dir`.
  ///
  /// \return The reader; NotFound when the directory holds no weight
  ///         shards, and the reader's own error text for malformed files.
  static StatusOr<CheckpointReader> Open(const std::string& dir);

  /// \brief A borrowed CPU tensor for `name`, or NotFound.
  StatusOr<Tensor> GetTensor(const std::string& name) const;

  /// \brief Every parameter name across all shards, in shard order.
  std::vector<std::string> Names() const;

 private:
  explicit CheckpointReader(std::vector<SafeTensorReader> shards)
      : shards_(std::move(shards)) {}

  std::vector<SafeTensorReader> shards_;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_SAFETENSORS_H_
