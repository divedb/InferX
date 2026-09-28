#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/models/checkpoint_config.h"
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

  /// \brief Converts a host tensor to a bfloat16 host tensor, owned.
  ///
  /// Works on slices of checkpoint tensors as well as whole tensors. Bf16
  /// inputs pass through unchanged (still borrowing their storage); f32/f16
  /// convert elementwise into a fresh allocation.
  static StatusOr<Tensor> AsHostBf16(const Tensor& host);

  /// \brief Host bf16 view of `name`, validated against the FULL tensor shape.
  ///
  /// Sharded loaders check the whole checkpoint tensor here, then slice the
  /// result to their rank's rows before uploading.
  ///
  /// \param name         Tensor name.
  /// \param expected_full Shape of the whole checkpoint tensor.
  /// \return             The host bf16 tensor, or an error status.
  StatusOr<Tensor> FindHostBf16(std::string_view name, const Shape& expected_full) const;

  /// \brief Uploads `name` to `device` as float32 (A_log/dt_bias style).
  StatusOr<Tensor> UploadF32(std::string_view name, const Shape& expected,
                             DeviceId device) const;

  /// \brief Dequantizes MXFP4-packed nibbles with E8M0 block scales to bf16.
  StatusOr<Tensor> DequantMxToBf16(const Tensor& blocks, const Tensor& scales,
                                   const Shape& logical) const;

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

/// \brief A checkpoint directory opened exactly once: the parsed config.json
///        and the mapped weights, ready for a family to translate.
struct LoadedCheckpoint {
  CheckpointConfig config;  ///< Parsed `config.json`.
  std::string config_json;  ///< Raw `config.json` text; families read their own
                            ///< fields (MoE, layer types) from here.
  Checkpoint weights;       ///< Mapped `.safetensors` shards.
};

/// \brief Reads and parses `config.json` of `dir`; weights stay unopened.
///
/// Identity is knowable from the config alone, so the registry rejects
/// unsupported architectures before any shard is mapped.
StatusOr<LoadedCheckpoint> LoadCheckpointConfig(const std::string& dir);

/// \brief Maps the `.safetensors` shards of `dir` into `checkpoint`.
Status OpenCheckpointWeights(LoadedCheckpoint& checkpoint, const std::string& dir);

/// \brief Reads `config.json` and maps the weights of `dir` in one pass.
///
/// One open serves the whole build: the selected family reads extra fields
/// from `config_json` and maps tensors out of `weights` without touching the
/// directory again.
StatusOr<LoadedCheckpoint> LoadCheckpoint(const std::string& dir);

}  // namespace inferx::models
