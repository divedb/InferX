/// \file
/// \brief The block: the one place transformer topology lives. A block is a
/// pre-norm residual unit — norm -> mixer -> add, norm -> feed-forward ->
/// add — with the mixer and feed-forward chosen per layer from variants.

#ifndef INFERX_MODELS_LAYERS_BLOCK_H_
#define INFERX_MODELS_LAYERS_BLOCK_H_

#include <variant>

#include "inferx/models/layers/feed_forward.h"
#include "inferx/models/layers/mixer.h"

namespace inferx::layers {

/// \brief RMS normalization parameters.
struct NormConfig {
  float eps = 1e-5f;
  bool plus_one = false;  ///< Scale by (1 + w) instead of w (Qwen3-Next).
};

/// \brief Configuration for one pre-norm block.
struct BlockConfig {
  NormConfig norm;  ///< Applied before the mixer and before the feed-forward.
  std::variant<AttentionConfig, GatedDeltaNetConfig> mixer;
  std::variant<SwiGluConfig, MoeConfig> feed_forward;
};

/// \brief Weights for one pre-norm block.
struct BlockWeights {
  Tensor input_norm;       ///< [hidden] mixer-side norm.
  Tensor post_mixer_norm;  ///< [hidden] feed-forward-side norm.
  AttentionWeights mixer;  ///< Recurrent mixers are not loadable yet.
  std::variant<SwiGluWeights, MoeWeights> feed_forward;
};

}  // namespace inferx::layers

#endif  // INFERX_MODELS_LAYERS_BLOCK_H_
