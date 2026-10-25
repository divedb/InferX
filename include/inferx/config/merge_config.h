#ifndef INFERX_CONFIG_MERGE_CONFIG_H_
#define INFERX_CONFIG_MERGE_CONFIG_H_

#include <set>

#include "inferx/config/generation_config.h"
#include "inferx/sampling/sampling_params.h"

namespace inferx {

/// \brief Merges generation defaults into `sampling`.
///
/// The defaults of GenerationConfig mirror SamplingParams, so a field the
/// checkpoint did not set assigns the sampling default — a no-op — and only
/// fields the file actually carries change anything. GenerationConfig::
/// max_new_tokens of 0 ("not constrained") never overrides the engine's
/// max_tokens default.
///
/// \param gen      Generation defaults to merge (LoadGenerationConfig's result).
/// \param sampling Sampling parameters to update in place.
/// \param skip     Field names whose caller-set values must survive the merge
///                 ("temperature", "top-p", "max-tokens" — the CLI flags a
///                 user passed explicitly, which win over the checkpoint).
void MergeGenerationConfig(const GenerationConfig& gen, sampling::SamplingParams* sampling,
                           const std::set<std::string>& skip = {});

}  // namespace inferx

#endif  // INFERX_CONFIG_MERGE_CONFIG_H_
