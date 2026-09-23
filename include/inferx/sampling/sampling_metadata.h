#ifndef INFERX_SAMPLING_SAMPLING_METADATA_H_
#define INFERX_SAMPLING_SAMPLING_METADATA_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/types/span.h"
#include "inferx/sampling/sampling_params.h"

namespace inferx::sampling {

/// \brief One scheduled request's resolved sampling knobs, in batch order.
///
/// The runtime form of SamplingParams: borrowed-pointer-free values the
/// sampler reads per row. `rng_offset` is the request's generated-token
/// count, used as the counter of the request's seeded RNG stream so draws
/// depend only on (seed, position), never on batch composition.
struct RequestSamplingParams {
  bool greedy = true;
  float temperature = 1.0f;
  std::uint32_t top_k = 0;
  float top_p = 1.0f;
  float min_p = 0.0f;
  float repetition_penalty = 1.0f;
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  bool has_logit_bias = false;
  bool has_allowed_token_ids = false;
  bool apply_penalties = false;  ///< Any repetition/presence/frequency effect.
  std::optional<std::uint64_t> seed;
  std::uint64_t rng_offset = 0;  ///< Generated tokens so far.
};

/// \brief The batch's sampling description, built per engine step.
///
/// Host-side today; the device mirror (packed SoA parameter tensors plus CSR
/// arrays for bias/allowlists and penalty token histories) is added together
/// with the fused kernel that consumes it, not before. `all_greedy` selects
/// the fast path that skips the pipeline entirely.
struct SamplingMetadata {
  /// \brief Builder input: the request's params plus its RNG position.
  struct PerRequest {
    const SamplingParams* params = nullptr;  ///< Borrowed for the Build call.
    std::uint64_t rng_offset = 0;            ///< Generated tokens so far.
  };

  int batch = 0;
  std::int64_t vocab_size = 0;
  bool all_greedy = true;
  std::vector<RequestSamplingParams> requests;  ///< Batch order.

  /// \brief Resolves `batch` (params + RNG positions) against a vocabulary.
  static SamplingMetadata Build(std::int64_t vocab_size,
                                absl::Span<const PerRequest> batch) {
    SamplingMetadata metadata;
    metadata.vocab_size = vocab_size;
    metadata.requests.reserve(batch.size());
    for (const PerRequest& per : batch) {
      const SamplingParams& p = *per.params;
      RequestSamplingParams r;
      r.greedy = p.IsGreedy();
      r.temperature = p.temperature;
      r.top_k = p.top_k;
      r.top_p = p.top_p;
      r.min_p = p.min_p;
      r.repetition_penalty = p.repetition_penalty;
      r.presence_penalty = p.presence_penalty;
      r.frequency_penalty = p.frequency_penalty;
      r.has_logit_bias = !p.logit_bias.empty();
      r.has_allowed_token_ids = !p.allowed_token_ids.empty();
      r.apply_penalties = p.repetition_penalty != 1.0f || p.presence_penalty != 0.0f ||
                          p.frequency_penalty != 0.0f;
      r.seed = p.seed;
      r.rng_offset = per.rng_offset;
      metadata.all_greedy = metadata.all_greedy && r.greedy;
      metadata.requests.push_back(r);
    }
    metadata.batch = static_cast<int>(batch.size());
    return metadata;
  }
};

}  // namespace inferx::sampling

#endif  // INFERX_SAMPLING_SAMPLING_METADATA_H_
