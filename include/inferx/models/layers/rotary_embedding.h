#ifndef INFERX_LAYERS_ROTARY_EMBEDDING_H_
#define INFERX_LAYERS_ROTARY_EMBEDDING_H_

#include "inferx/ops/rotary.h"

namespace inferx {
namespace models {

/// \brief Rotary position embedding: frozen parameters, no buffers.
///
/// Frequencies are computed inside the kernels from the params (see
/// `ops::InverseFrequency`), so unlike vLLM this module holds no cos/sin
/// cache. It exists to give models one named owner of the rope geometry and
/// to carry it through configs and weight bindings readably.
class RotaryEmbedding {
 public:
  RotaryEmbedding() = default;

  /// \brief Wraps rope parameters validated by the op layer.
  explicit RotaryEmbedding(ops::RotaryParams params) : params_(params) {}

  const ops::RotaryParams& Params() const { return params_; }
  int64_t RotaryDim() const { return params_.rotary_dim; }

 private:
  ops::RotaryParams params_;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_LAYERS_ROTARY_EMBEDDING_H_
