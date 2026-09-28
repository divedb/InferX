#pragma once

#include <memory>
#include <string_view>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/models/causal/decoder_config.h"
#include "inferx/models/checkpoint.h"
#include "inferx/models/checkpoint_config.h"
#include "inferx/models/model.h"
#include "nlohmann/json.hpp"

namespace inferx {

/// Checkpoint identity and its factory. Dense factories are generated from
/// model traits; other architectures can provide their own Model builder.
struct Family {
  std::string_view model_type;
  std::string_view architecture;
  StatusOr<std::unique_ptr<Model>> (*build)(const nlohmann::json& config,
                                            models::LoadedCheckpoint& checkpoint,
                                            DeviceId device, int max_tokens, int max_seqs,
                                            const ParallelConfig& parallel) = nullptr;
  // Optional configuration-only inspection for decoder-based families.
  StatusOr<causal::DecoderConfig> (*translate)(const nlohmann::json& config) = nullptr;
};

/// \brief The family a checkpoint's `model_type` / `architectures` selects.
///
/// Resolution touches nothing but the parsed config, so callers reject
/// unknown architectures before opening any weight file.
StatusOr<const Family*> ResolveFamily(const CheckpointConfig& config);

/// \brief Builds the model `family` describes from an opened checkpoint.
///
/// Dense factories share translation, validation, loading, and assembly.
/// Custom factories own their model construction.
StatusOr<std::unique_ptr<Model>> BuildFamily(const Family& family,
                                             models::LoadedCheckpoint& checkpoint,
                                             DeviceId device, int max_tokens, int max_seqs,
                                             const ParallelConfig& parallel = {});

/// \brief Translates `config_json` with the family `identity` selects.
///
/// Parses, invokes the family translator, and validates the result.
StatusOr<causal::DecoderConfig> TranslateFamilyConfig(const CheckpointConfig& identity,
                                                      std::string_view config_json);

/// \brief Resolves the family and builds, in one call.
StatusOr<std::unique_ptr<Model>> BuildModel(models::LoadedCheckpoint& checkpoint,
                                            DeviceId device, int max_tokens, int max_seqs,
                                            const ParallelConfig& parallel = {});

}  // namespace inferx
