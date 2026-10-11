#ifndef INFERX_MODELS_LOAD_OPTIONS_H_
#define INFERX_MODELS_LOAD_OPTIONS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/models/model.h"

namespace inferx {
namespace models {

/// \brief Execution dtype request, with a follow-the-checkpoint default.
enum class DType : std::uint8_t {
  kAuto,       ///< Execute in the checkpoint's own dtype (converted if the
               ///< kernels require another).
  kBFloat16,   ///< Bfloat16 weights and activations.
  kFloat16,    ///< Not wired into the rebuilt model layer yet.
  kFloat32,    ///< Not wired into the rebuilt model layer yet.
};

/// \brief How a quantization_config in config.json is treated.
enum class QuantizationPolicy : std::uint8_t {
  /// Reads `quantization_config`; a quantized checkpoint currently fails
  /// with Unimplemented until the dequantization path is restored.
  kAuto,
  /// Ignores the field and treats the checkpoint as dense.
  kNone,
};

/// \brief Strictness and ergonomics knobs of the weight load.
struct LoadPolicy {
  /// Missing or unexpected checkpoint tensors are errors (the default);
  /// when false, unexpected tensors are skipped with a warning instead.
  bool strict_matching = true;

  /// Logs a summary of the load (tensors consumed, bytes staged).
  bool report_progress = false;

  /// Skips reading the checkpoint and zero-fills every bound tensor --
  /// shape and routing tests, and capture-time dry runs, need no bytes.
  bool dummy_weights = false;
};

/// \brief Everything `LoadModel` needs: where the checkpoint is, what to
///        build, and how strictly to match it.
struct ModelLoadOptions {
  /// Directory holding config.json and .safetensors files.
  std::string directory;

  /// Device this rank's weights live on.
  Device device;

  /// Execution dtype; kAuto follows the checkpoint's own dtype.
  DType dtype = DType::kAuto;

  /// kAuto reads `quantization_config` from config.json.
  QuantizationPolicy quantization = QuantizationPolicy::kAuto;

  /// This rank's tensor-parallel position; the default shards nothing.
  ParallelConfig parallel;

  /// Strictness of tensor matching, progress reporting, dummy weights.
  LoadPolicy policy;

  /// \brief Checks that can be validated before any file is touched.
  Status Validate() const;

  /// \brief Parses a CLI dtype string; empty means kAuto.
  static StatusOr<DType> ParseDType(const std::string& text);
};

/// \brief Builds the model for `options.parallel`'s rank and loads its
///        weights per `options.policy`.
///
/// Thin composition over the registry (architecture resolution, dtype
/// resolution) and `Model::LoadWeights`; see `LoadPolicy` for the
/// non-default paths.
StatusOr<std::unique_ptr<Model>> LoadModel(const ModelLoadOptions& options);

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_LOAD_OPTIONS_H_
