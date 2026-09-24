#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/models/safe_tensors_reader.h"

namespace inferx::models {

/// \brief A .safetensors checkpoint directory, memory-mapped.
///
/// Owns one SafeTensorReader per shard: a single model.safetensors, or every
/// *.safetensors file of a sharded checkpoint (auxiliary files -- optimizers,
/// adapters -- are skipped). Tensors stay zero-copy views over the mapping
/// and keep their checkpoint dtype, so a bf16 checkpoint uploads with one
/// host-to-device copy and no conversion pass.
///
/// Lifetime: tensors returned by Find() borrow the mappings this object
/// owns. Keep the Checkpoint alive as long as any of them is used.
class Checkpoint {
 public:
  /// \brief Maps `model.safetensors` in `dir`, or every shard of a sharded
  ///        checkpoint found there.
  ///
  /// \param dir Model directory containing the checkpoint.
  /// \return    The checkpoint, or an error status.
  static StatusOr<Checkpoint> Open(const std::string& dir);

  /// \brief True if `name` names a tensor in any shard.
  bool Contains(std::string_view name) const;

  /// \brief Zero-copy host view of `name` in its checkpoint dtype.
  ///
  /// \param name Tensor name.
  /// \return     The tensor, or nullopt if no shard has it.
  std::optional<Tensor> Find(std::string_view name) const;

  /// \brief Uploads `name` to `device` as bfloat16.
  ///
  /// \param name     Tensor name.
  /// \param expected Shape the caller requires; a mismatch is an error.
  /// \param device   Destination device.
  /// \return         The device tensor, or an error status. bf16 checkpoints
  ///                 copy straight through; f32/f16 are converted on the host
  ///                 first.
  StatusOr<Tensor> UploadBf16(std::string_view name, const Shape& expected,
                              DeviceId device) const;

 private:
  std::vector<SafeTensorReader> shards_;
};

}  // namespace inferx::models
