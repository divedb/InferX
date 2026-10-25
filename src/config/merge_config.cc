#include "inferx/config/merge_config.h"

namespace inferx {

void MergeGenerationConfig(const GenerationConfig& gen, sampling::SamplingParams* sampling,
                           const std::set<std::string>& skip) {
  // Unconditional assignment is safe: GenerationConfig defaults mirror
  // SamplingParams, so fields the checkpoint did not set reassign the
  // sampling defaults (a no-op). top_k arrives pre-normalized (HF -1 → 0).
  if (!skip.count("temperature")) sampling->temperature = gen.temperature;
  if (!skip.count("top-p")) sampling->top_p = gen.top_p;
  sampling->min_p = gen.min_p;
  sampling->top_k = gen.top_k > 0 ? static_cast<std::uint32_t>(gen.top_k) : 0;
  sampling->repetition_penalty = gen.repetition_penalty;
  if (!skip.count("max-tokens") && gen.max_new_tokens > 0) {
    sampling->max_tokens = static_cast<std::uint32_t>(gen.max_new_tokens);
  }
}

}  // namespace inferx
