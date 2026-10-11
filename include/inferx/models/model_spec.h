#ifndef INFERX_MODELS_MODEL_SPEC_H_
#define INFERX_MODELS_MODEL_SPEC_H_

#include <cstdint>
#include <string>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/models/requirements.h"
#include "inferx/tokenizer/id.h"

namespace inferx {
namespace models {

/// \brief Engine-facing facts about one loaded model.
///
/// This is the narrow waist between the model layer and the runtime: the
/// scheduler, executor, and serving layer see a model only through this
/// description plus the ModelRunner that owns it. Family-specific fields
/// stay inside the family's config; anything the runtime must know to
/// schedule, size caches, or sample belongs here.
struct ModelSpec {
  /// Checkpoint architecture name, e.g. "Qwen3ForCausalLM".
  std::string architecture;

  /// Rows of the embedding table and the LM head; after vocab curation this
  /// is the id space logits are indexed by.
  int64_t vocab_size = 0;

  /// Width of the residual stream.
  int64_t hidden_size = 0;

  /// Number of decoder layers.
  int64_t num_layers = 0;

  /// Checkpoint context limit (max_position_embeddings).
  int64_t max_position_embeddings = 0;

  /// Tokens that end generation; sampled ids are checked against these.
  std::vector<TokenId> eos_token_ids;

  /// True when the LM head shares the embedding table's weights.
  bool tie_word_embeddings = false;

  /// Element type of weights and activations.
  DataType dtype = DataType::kBFloat16;

  /// Device the weights (and this rank's execution) live on.
  Device device;

  /// Persistent per-layer state the runner must allocate.
  StateRequirements state;
};

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_MODEL_SPEC_H_
