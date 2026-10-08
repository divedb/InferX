/// \file
/// \brief Architecture registry: maps checkpoint identity to a typed model
/// factory. The registry erases the concrete configuration and model types
/// at the factory boundary; inside a factory, construction stays fully
/// typed. Adding a model never extends a global configuration variant.

#ifndef INFERX_MODELS_REGISTRY_H_
#define INFERX_MODELS_REGISTRY_H_

#include <concepts>
#include <memory>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/models/load_options.h"
#include "inferx/models/model.h"
#include "inferx/models/model_info.h"
#include "inferx/util/config_view.h"

namespace inferx::models {

class CausalLanguageModel;

/// \brief One registered architecture behind a type-erased factory.
///
/// `build` performs the full lifecycle for its architecture:
///   1. parse and validate the architecture config from `config`,
///   2. validate execution support for the requested device, dtype, and
///      parallelism,
///   3. construct the model structure and declare parameters and state,
///   4. load and validate weights,
///   5. initialize execution resources.
struct ArchitectureEntry {
  std::string_view architecture;  ///< e.g. "Qwen3ForCausalLM".
  std::string_view model_type;    ///< e.g. "qwen3"; may be empty.
  StatusOr<std::unique_ptr<CausalLanguageModel>> (*build)(
      const ConfigView& config, const ModelLoadOptions& options) = nullptr;
};

/// \brief Registry of known architectures. Lookup is by checkpoint identity
///        (`model_type` then `architectures[0]`); both matching different
///        entries is an ambiguity error, never a silent choice.
class ArchitectureRegistry {
 public:
  /// \brief Registers one entry; rejects entries whose identity (either
  ///        field) collides with an existing one.
  Status Register(ArchitectureEntry entry);

  /// \brief The builtin architectures, explicitly registered for
  ///        predictable static linking.
  static const ArchitectureRegistry& Builtins();

  /// \brief Resolves the entry a checkpoint's identity selects.
  ///
  /// \param model_type    The config's `model_type`; empty skips it.
  /// \param architecture  The config's `architectures[0]`; empty skips it.
  /// \return              The entry, or NotFound listing supported names.
  StatusOr<const ArchitectureEntry*> Resolve(std::string_view model_type,
                                             std::string_view architecture) const;

  const std::vector<ArchitectureEntry>& entries() const { return entries_; }

 private:
  std::vector<ArchitectureEntry> entries_;
};

/// \brief Requirements a model implementation satisfies to register:
///        a nested `Config` type with `Parse(const ConfigView&)`, and a
///        static `Build(const Config&, const ModelLoadOptions&)` factory.
template <class ModelT>
concept ArchitectureFactory = requires {
  typename ModelT::Config;
  requires std::same_as<
      decltype(ModelT::Build(std::declval<const typename ModelT::Config&>(),
                             std::declval<const ModelLoadOptions&>())),
      StatusOr<std::unique_ptr<ModelT>>>;
};

/// \brief Reports a fatal builtin registration conflict (aborts).
void FatalRegistryError(std::string_view architecture, const Status& status);

/// \brief Registers `ModelT` under `architecture` and `model_type`,
///        erasing the typed factory behind ArchitectureEntry::build.
///
/// The erased factory parses the architecture config, then hands the typed
/// config and load options to ModelT::Build. Parse failures surface with
/// the full config key path.
template <class ModelT>
  requires ArchitectureFactory<ModelT>
void RegisterArchitecture(ArchitectureRegistry& registry, std::string_view architecture,
                          std::string_view model_type) {
  ArchitectureEntry entry;
  entry.architecture = architecture;
  entry.model_type = model_type;
  entry.build = [](const ConfigView& config,
                   const ModelLoadOptions& options)
      -> StatusOr<std::unique_ptr<CausalLanguageModel>> {
    using Config = typename ModelT::Config;
    INFERX_ASSIGN_OR_RETURN(auto parsed, Config::Parse(config));
    INFERX_ASSIGN_OR_RETURN(auto model, ModelT::Build(parsed, options));
    return std::unique_ptr<CausalLanguageModel>(std::move(model));
  };
  // Built-in identities are programmer errors; fail fast and loudly.
  const Status registered = registry.Register(std::move(entry));
  if (!registered.ok()) {
    FatalRegistryError(architecture, registered);
  }
}

}  // namespace inferx::models

#endif  // INFERX_MODELS_REGISTRY_H_
