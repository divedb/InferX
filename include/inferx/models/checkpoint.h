#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/models/checkpoint_config.h"
#include "inferx/models/quant_config.h"
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

  /// \brief The checkpoint's weight-quantization plan, if it has one.
  const std::optional<QuantPlan>& quant_plan() const { return quant_; }

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

  /// \brief Dequantizes GPTQ-packed weights to a bf16 host tensor.
  ///
  /// Layout, validated against `logical` [out, in]: qweight [in / pack, out]
  /// packs `pack` input values per int32 word, least-significant value
  /// first; qzeros [in / group_size, out / pack] packs output values the
  /// same way; scales [in / group_size, out] in f16, bf16, or f32; g_idx
  /// maps each input to its group (nullopt means static in / group_size
  /// grouping). Values reconstruct as w = scale * (q - zero - 1): stored
  /// zero points carry a +1 bias, and a symmetric int8 checkpoint's 0x7F
  /// bytes fold into a net offset of 2^(bits-1).
  StatusOr<Tensor> DequantGptq(const Tensor& qweight, const Tensor& qzeros,
                               const Tensor& scales, const std::optional<Tensor>& g_idx,
                               const QuantPlan& plan, const Shape& logical) const;

  /// \brief Dequantizes AWQ (gemm version) weights to a bf16 host tensor.
  ///
  /// Layout, validated against `logical` [out, in]: qweight [in, out / pack]
  /// and qzeros [in / group_size, out / pack] both pack output values per
  /// int32 word, least-significant value first; scales [in / group_size,
  /// out]. Values reconstruct as w = scale * (q - zero + 1): AWQ zero
  /// points store a +1 bias relative to their nominal value.
  StatusOr<Tensor> DequantAwq(const Tensor& qweight, const Tensor& qzeros,
                              const Tensor& scales, const QuantPlan& plan,
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
  /// \brief Rebuilds the bf16 [out, in] weight behind a `.weight` name from
  ///        its packed GPTQ/AWQ companions.
  ///
  /// The fallback of FindHostBf16 for quantized checkpoints: dense tensors
  /// win when present (norms, embeddings, and unquantized projections in a
  /// partially quantized checkpoint all take the plain path), and a missing
  /// dense tensor without its packed companions reports the dense name.
  StatusOr<Tensor> FindQuantizedBf16(std::string_view name, const Shape& logical) const;

  std::optional<QuantPlan> quant_;
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
