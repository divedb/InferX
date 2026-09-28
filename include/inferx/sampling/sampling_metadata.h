#ifndef INFERX_SAMPLING_SAMPLING_METADATA_H_
#define INFERX_SAMPLING_SAMPLING_METADATA_H_

#include <cstdint>
#include <optional>
#include <vector>

#include <cstring>
#include <map>

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
    /// Prompt plus generated tokens, for penalties; borrowed for Build.
    const std::vector<std::int32_t>* history = nullptr;
  };

  int batch = 0;
  std::int64_t vocab_size = 0;
  bool all_greedy = true;
  std::vector<RequestSamplingParams> requests;  ///< Batch order.

  /// \brief Device-mirror payloads, built by Build(): CSR entries packed
  ///        alongside the per-row knobs so one async upload serves the batch.
  ///
  /// Bias entries pack (token, float bits) into uint64; history entries pack
  /// (token, occurrence count) the same way. The engine seed stands in for
  /// requests that did not set one.
  struct DevicePayload {
    std::vector<float> temperature, top_p, min_p, penalties;
    std::vector<int32_t> top_k, greedy;
    std::vector<uint64_t> seeds, rng_offsets;
    std::vector<int32_t> bias_ptr{0}, allow_ptr{0}, hist_ptr{0};
    std::vector<uint64_t> bias_entries, hist_entries;
    std::vector<int32_t> allow_entries;
  };
  DevicePayload device;
  uint64_t engine_seed = 0;

  /// \brief Resolves `batch` (params + RNG positions + histories) against a
  ///        vocabulary, also packing the device-mirror CSR payloads.
  static SamplingMetadata Build(std::int64_t vocab_size,
                                absl::Span<const PerRequest> batch,
                                uint64_t engine_seed = 0) {
    SamplingMetadata metadata;
    metadata.vocab_size = vocab_size;
    metadata.engine_seed = engine_seed;
    metadata.requests.reserve(batch.size());
    auto& d = metadata.device;
    d.temperature.reserve(batch.size());
    d.top_k.reserve(batch.size());
    d.greedy.reserve(batch.size());
    d.seeds.reserve(batch.size());
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

      d.temperature.push_back(p.temperature);
      d.top_k.push_back(static_cast<int32_t>(p.top_k));
      d.top_p.push_back(p.top_p);
      d.min_p.push_back(p.min_p);
      d.penalties.push_back(p.repetition_penalty);
      d.penalties.push_back(p.presence_penalty);
      d.penalties.push_back(p.frequency_penalty);
      d.greedy.push_back(r.greedy ? 1 : 0);
      d.seeds.push_back(p.seed.value_or(engine_seed));
      d.rng_offsets.push_back(per.rng_offset);
      for (const auto& [token, bias] : p.logit_bias) {
        uint32_t bits = 0;
        std::memcpy(&bits, &bias, sizeof(bits));
        d.bias_entries.push_back((uint64_t(static_cast<uint32_t>(token)) << 32) | bits);
      }
      d.bias_ptr.push_back(static_cast<int32_t>(d.bias_entries.size()));
      for (int32_t token : p.allowed_token_ids) d.allow_entries.push_back(token);
      d.allow_ptr.push_back(static_cast<int32_t>(d.allow_entries.size()));
      if (per.history != nullptr) {
        // Deduplicate with counts: presence fires once, frequency scales.
        std::map<int32_t, int32_t> counts;
        for (int32_t token : *per.history) counts[token]++;
        for (const auto& [token, count] : counts) {
          d.hist_entries.push_back((uint64_t(static_cast<uint32_t>(token)) << 32) |
                                   uint32_t(count));
        }
      }
      d.hist_ptr.push_back(static_cast<int32_t>(d.hist_entries.size()));
    }
    metadata.batch = static_cast<int>(batch.size());
    return metadata;
  }
};

}  // namespace inferx::sampling

#endif  // INFERX_SAMPLING_SAMPLING_METADATA_H_
