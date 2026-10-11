#ifndef INFERX_MODELS_REGISTRY_H_
#define INFERX_MODELS_REGISTRY_H_

#include <memory>
#include <string>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/dtype.h"
#include "inferx/core/status.h"
#include "inferx/models/load_options.h"
#include "inferx/models/model.h"
#include "inferx/util/hf_config.h"

namespace inferx {
namespace models {

/// \brief Inputs every registered architecture builds from.
struct ModelCreateParams {
  const HfConfig& config;   ///< Parsed config.json.
  Device device;          ///< Where weights and activations live.
  const ParallelConfig& parallel;  ///< This rank's sharding geometry.
};

/// \brief Builds one architecture's model, allocated and unloaded.
using ModelFactory = StatusOr<std::unique_ptr<Model>> (*)(const ModelCreateParams&);

/// \brief One registered architecture: its HF names and its factory.
struct ArchitectureEntry {
  const char* architecture;  ///< `architectures[0]`, e.g. "Qwen3ForCausalLM".
  const char* model_type;    ///< `model_type`, e.g. "qwen3".
  ModelFactory create;
};

/// \brief The static table of architectures this build can run.
///
/// Adding a family means implementing its classes and adding one row here;
/// no shared code changes. Resolution prefers `architectures[0]` (exact,
/// versioned) and falls back to `model_type` (families that rename their
/// CausalLM class stay loadable).
class ArchitectureRegistry {
 public:
  /// \brief The compile-time registry: Qwen3, Qwen3-MoE, Llama.
  static const ArchitectureRegistry& Builtins();

  /// \brief Finds the entry for a checkpoint.
  ///
  /// \param model_type    The config's `model_type` ("" when absent).
  /// \param architecture  The config's `architectures[0]` ("" when absent).
  /// \return              The entry, or NotFound naming what was tried.
  StatusOr<const ArchitectureEntry*> Resolve(const std::string& model_type,
                                             const std::string& architecture) const;

  const std::vector<ArchitectureEntry>& Entries() const { return entries_; }

 private:
  explicit ArchitectureRegistry(std::vector<ArchitectureEntry> entries)
      : entries_(std::move(entries)) {}

  std::vector<ArchitectureEntry> entries_;
};

/// \brief Resolves an execution-dtype request against a checkpoint.
///
/// kAuto follows the checkpoint's `torch_dtype`; the kernels execute
/// bfloat16, so every supported request lands there (conversion happens at
/// load) and the others report Unimplemented.
StatusOr<DataType> ResolveDType(DType requested, const HfConfig& config);

/// \brief Builds the model a checkpoint directory describes, unloaded.
///
/// Reads config.json, resolves the architecture and dtype, and returns the
/// allocated model (weights are loaded by `LoadWeights`/`LoadModel` next).
/// Sets the spec's device so the load path can find its runtime.
StatusOr<std::unique_ptr<Model>> BuildModel(const ModelLoadOptions& options);

}  // namespace models
}  // namespace inferx

#endif  // INFERX_MODELS_REGISTRY_H_
