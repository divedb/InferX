#ifndef INFERX_SAMPLING_SAMPLING_PARAMS_H_
#define INFERX_SAMPLING_SAMPLING_PARAMS_H_

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "inferx/core/status.h"

namespace inferx::sampling {

/// \brief Request-level sampling configuration: the API-facing type users
///        construct, independent of CLI11, the engine, and any device.
///
/// One struct with several readers. Distribution-shaping fields (temperature,
/// top-k/top-p/min-p, logit bias, penalties) are consumed by the sampler;
/// termination fields (max/min tokens, stop tokens/strings, ignore_eos) are
/// consumed by the scheduler; `seed` feeds the sampler's per-request RNG.
/// They are kept together because they travel together: every front end
/// expresses a generation request as one SamplingParams value.
///
/// Token ids are plain int32_t: this header sits below the engine and must
/// not depend on engine::TokenId.
struct SamplingParams {
  /// Temperature applied to logits before sampling.
  ///
  /// 0 enables greedy decoding (the argmax fast path); 1 applies no scaling.
  /// Must be >= 0.
  float temperature = 1.0f;

  /// Number of highest-probability tokens retained. 0 disables the
  /// filter; 1 is greedy decoding. Applied on logits, before softmax.
  std::uint32_t top_k = 0;

  /// \brief Nucleus (top-p) cumulative probability threshold, applied on
  ///        probabilities. 1 disables the filter. Must be in (0, 1].
  float top_p = 1.0f;

  /// \brief Minimum probability relative to the most probable token,
  ///        applied on probabilities. 0 disables the filter. Must be in
  ///        [0, 1].
  float min_p = 0.0f;

  /// \brief Additive per-token logit bias, applied before penalties and
  ///        temperature. A bias of -INFINITY bans the token; a sparse
  ///        {token: +INFINITY} choice is an allowlist of one.
  std::map<std::int32_t, float> logit_bias;

  /// \brief Explicit allowlist; empty means the whole vocabulary. Conceptual
  ///        output constraints (grammars, structured decoding) compile to
  ///        this.
  std::vector<std::int32_t> allowed_token_ids;

  /// \brief Multiplicative penalty on tokens already present in the prompt or
  ///        output. 1 disables it. Must be > 0.
  float repetition_penalty = 1.0f;

  /// \brief Constant penalty on tokens already present. 0 disables it.
  ///        Must be in [-2, 2].
  float presence_penalty = 0.0f;

  /// \brief Penalty proportional to occurrence counts. 0 disables it.
  ///        Must be in [-2, 2].
  float frequency_penalty = 0.0f;

  /// \brief Optional per-request RNG seed. Counter-based GPU sampling derives
  ///        each draw from (seed, generated-token index), so a seed is
  ///        reproducible under re-batching. Absent means engine-chosen.
  std::optional<std::uint64_t> seed;

  /// \brief Number of log-probabilities to return per generated token
  ///        (0 = none): the sampled token's and up to this many runners-up.
  std::uint32_t logprobs = 0;

  /// \brief Number of prompt log-probabilities to return (0 = none).
  std::uint32_t prompt_logprobs = 0;

  /// \brief Maximum number of generated tokens (prompt excluded).
  std::uint32_t max_tokens = 16;

  /// \brief Minimum tokens before normal stop conditions may fire.
  std::uint32_t min_tokens = 0;

  /// \brief True when EOS must not terminate generation.
  bool ignore_eos = false;

  /// \brief Additional token ids that terminate generation.
  std::vector<std::int32_t> stop_token_ids;

  /// \brief Text sequences that terminate generation.
  std::vector<std::string> stop_strings;

  /// \brief True when the distribution-shaping fields reduce to argmax and
  ///        the sampler can take its greedy fast path.
  bool IsGreedy() const { return temperature == 0.0f || top_k == 1; }

  /// \brief Validates field ranges and invariants (NaN-safe).
  ///
  /// Vocab-relative checks (token ids in range, top-k <= vocab) happen at the
  /// layer that knows the vocabulary.
  Status Validate() const;
};

}  // namespace inferx::sampling

#endif  // INFERX_SAMPLING_SAMPLING_PARAMS_H_
