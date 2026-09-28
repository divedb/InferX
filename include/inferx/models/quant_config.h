#ifndef INFERX_MODELS_QUANT_CONFIG_H_
#define INFERX_MODELS_QUANT_CONFIG_H_

#include <cstdint>
#include <optional>
#include <string>

#include "inferx/core/status.h"

namespace inferx::models {

/// \brief Weight-only quantization plan parsed from a checkpoint's
///        `quantization_config` section of `config.json`.
///
/// The engine dequantizes at load: the plan names the packing format and
/// its parameters so Checkpoint can rebuild bf16 weights from the packed
/// tensors (qweight/qzeros/scales, plus g_idx for GPTQ) before any loader
/// looks at them. Every downstream component -- parallel linears, device
/// upload, kernels -- keeps seeing ordinary bf16 weights.
struct QuantPlan {
  enum class Format { kGptq, kAwq };

  /// \brief Packing format of the quantized linear weights.
  Format format = Format::kGptq;

  /// \brief Quantization bit width (4 or 8).
  int bits = 4;

  /// \brief Input values sharing one scale and zero point.
  std::int64_t group_size = 128;

  /// \brief Parses the plan out of a `config.json` text.
  ///
  /// \return The plan when linear weights are quantized in a format this
  ///         engine can dequantize, nullopt for dense checkpoints (and for
  ///         formats handled elsewhere, like gpt-oss's fused MXFP4
  ///         experts), or an error for quantization it cannot load -- a
  ///         clear failure instead of missing-tensor noise later.
  static StatusOr<std::optional<QuantPlan>> FromConfigJson(const std::string& json);
};

}  // namespace inferx::models

#endif  // INFERX_MODELS_QUANT_CONFIG_H_
