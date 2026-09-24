#pragma once

#include <memory>
#include <string_view>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/logging.h"
#include "inferx/core/status.h"
#include "inferx/engine/parallel_config.h"
#include "inferx/models/causal/decoder_stack.h"
#include "inferx/models/causal/weight_mapping.h"
#include "inferx/models/checkpoint.h"
#include "inferx/models/checkpoint_config.h"
#include "inferx/models/model.h"
#include "nlohmann/json.hpp"

namespace inferx {

/// \brief One architecture family: how a checkpoint identity maps to a Model,
///        expressed at the lowest tier that fits.
///
/// Tier 1 -- knobs only, no function pointers: the standard dense causal
/// decoder whose config.json differs just by the flags below and its
/// checkpoint names. Tier 2 -- `translate`: the same standard decoder behind
/// custom config translation (MoE fields, layer types). Tier 3 -- `build`: a
/// genuinely different executor graph (encoder-decoder, multimodal towers,
/// native SSM) that owns everything, including its own Model subclass.
///
/// The rule that keeps this struct small: never add a flag for a second
/// family's sake -- give that family `translate`, and reach for `build` only
/// when no DecoderConfig can express the model.
struct Family {
  std::string_view model_type;    ///< HF `model_type`, e.g. "llama".
  std::string_view architecture;  ///< HF `architectures[0]`, e.g. "LlamaForCausalLM".

  /// \brief Tier 3: owns the whole build for a structurally different model.
  StatusOr<std::unique_ptr<Model>> (*build)(const nlohmann::json& config,
                                            models::LoadedCheckpoint& checkpoint,
                                            DeviceId device, int max_tokens, int max_seqs,
                                            const ParallelConfig& parallel) = nullptr;

  /// \brief Tier 2: custom config translation onto the standard decoder.
  StatusOr<causal::DecoderConfig> (*translate)(const nlohmann::json& config) = nullptr;

  bool qk_norm = false;                ///< Tier 1: per-head q/k RMSNorm (Qwen-style).
  bool plus_one_norm = false;          ///< Tier 1: Gemma-style (1 + x) norms.
  causal::CheckpointLayout layout{};   ///< Tier 1: checkpoint tensor names.
};

/// \brief Registry rows families added at static init.
///
/// Inline with function-local storage so a registering translation unit
/// never depends on registry code elsewhere in the link: the family archive
/// is whole-archived, and its members must reference nothing outside
/// themselves, headers, and libraries that follow it (see the CMake LINK
/// RULE on inferx_model_families).
inline std::vector<Family>& RegisteredFamilies() {
  static std::vector<Family> registry;
  return registry;
}

/// \brief Adds a family from its own translation unit, once per identity it
///        claims, at static init (see models/qwen3.cc).
///
/// Registration order never matters. Aborts on a duplicate model_type among
/// registered families -- a collision is a programming error.
inline void RegisterFamily(Family family) {
  for (const auto& existing : RegisteredFamilies()) {
    if (existing.model_type == family.model_type) {
      INFERX_LOG(FATAL) << "duplicate model family registration for '"
                        << std::string(family.model_type) << "'";
    }
  }
  RegisteredFamilies().push_back(std::move(family));
}

/// \brief The family a checkpoint's `model_type` / `architectures` selects.
///
/// Resolution touches nothing but the parsed config, so callers reject
/// unknown architectures before opening any weight file.
StatusOr<const Family*> ResolveFamily(const CheckpointConfig& config);

/// \brief Builds the model `family` describes from an opened checkpoint.
///
/// The one composition path: tier-3 build functions own everything; tiers 1
/// and 2 translate, validate, and assemble through BuildCausalLM, so no
/// family repeats that plumbing.
StatusOr<std::unique_ptr<Model>> BuildFamily(const Family& family,
                                             models::LoadedCheckpoint& checkpoint,
                                             DeviceId device, int max_tokens, int max_seqs,
                                             const ParallelConfig& parallel = {});

/// \brief Translates `config_json` with the family `identity` selects.
///
/// The config-only seam family tests drive: parse, tier dispatch (family
/// translator or the standard knobs), then validation.
StatusOr<causal::DecoderConfig> TranslateFamilyConfig(const CheckpointConfig& identity,
                                                      std::string_view config_json);

/// \brief Resolves the family and builds, in one call.
StatusOr<std::unique_ptr<Model>> BuildModel(models::LoadedCheckpoint& checkpoint,
                                            DeviceId device, int max_tokens, int max_seqs,
                                            const ParallelConfig& parallel = {});

}  // namespace inferx
