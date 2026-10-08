#ifndef INFERX_MODELS_MODEL_INFO_H_
#define INFERX_MODELS_MODEL_INFO_H_

#include <cstdint>
#include <string>

namespace inferx::models {

/// \brief Capabilities a model exposes to the engine.
enum class ModelTask : uint8_t {
  kCausalLm,  ///< Next-token prediction over a flat token batch.
};

/// \brief Engine-facing model facts: vocabulary, context limit, stop ids,
///        and the task this model executes. Nothing here describes internal
///        layer geometry; ModelRequirements carries execution needs.
struct ModelInfo {
  std::string architecture;             ///< Checkpoint identity, e.g. "Qwen3ForCausalLM".
  int64_t vocab_size = 0;               ///< Full vocabulary width across all ranks.
  int64_t max_position_embeddings = 0;  ///< Supported context length.
  int64_t bos_token_id = -1;            ///< Beginning-of-sequence id; -1 absent.
  int64_t eos_token_id = -1;            ///< Stop-generation id; -1 absent.
  ModelTask task = ModelTask::kCausalLm;
};

}  // namespace inferx::models

#endif  // INFERX_MODELS_MODEL_INFO_H_
