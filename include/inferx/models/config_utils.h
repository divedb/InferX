#ifndef INFERX_MODELS_CONFIG_UTILS_H_
#define INFERX_MODELS_CONFIG_UTILS_H_

#include "inferx/core/status.h"
#include "inferx/ops/rotary.h"
#include "inferx/util/config_view.h"

namespace inferx {
namespace models {

/// \brief Finalizes a checkpoint's `rope_scaling` object into kernel
///        parameters.
///
/// The JSON carries raw factors and bounds; the per-frequency formulas the
/// kernels run (see `ops::InverseFrequency`) need the YaRN ramp expressed
/// in frequency indices and the attention temperature folded in. Computing
/// those here, once on the host, keeps one rope formula everywhere. An
/// absent object yields unscaled rope.
///
/// \param scaling    View of the `rope_scaling` object.
/// \param rotary_dim Rotated columns per head.
/// \param theta      Rope base frequency.
StatusOr<ops::RopeScaling> ParseRopeScaling(const std::optional<ConfigView>& scaling,
                                            int64_t rotary_dim, double theta);

/// \brief Builds rope parameters from the common HF fields.
///
/// \param text         The model's text config view.
/// \param rotary_dim   Rotated columns per head (usually head_dim).
/// \param default_theta Base frequency when `rope_theta` is absent.
StatusOr<ops::RotaryParams> ParseRope(const ConfigView& text, int64_t rotary_dim,
                                      double default_theta);

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_CONFIG_UTILS_H_
