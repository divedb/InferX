#include "inferx/models/registry.h"

#include <utility>

#include "inferx/core/logging.h"
#include "inferx/models/llama/llama_causal_lm.h"
#include "inferx/models/qwen3/qwen3_causal_lm.h"
#include "inferx/models/qwen3_moe/qwen3_moe_causal_lm.h"

namespace inferx {
namespace models {

namespace {

template <typename FamilyConfig, typename FamilyModel>
StatusOr<std::unique_ptr<Model>> BuildFamily(const ModelCreateParams& params) {
  INFERX_ASSIGN_OR_RETURN(FamilyConfig config, FamilyConfig::FromHf(params.config));
  return FamilyModel::Create(config, params.parallel, DataType::kBFloat16,
                             params.device);
}

}  // namespace

const ArchitectureRegistry& ArchitectureRegistry::Builtins() {
  static const ArchitectureRegistry* registry = new ArchitectureRegistry({
      {"Qwen3ForCausalLM", "qwen3", &BuildFamily<qwen3::Qwen3Config, qwen3::Qwen3ForCausalLM>},
      {"Qwen3MoeForCausalLM", "qwen3_moe",
       &BuildFamily<qwen3_moe::Qwen3MoeConfig, qwen3_moe::Qwen3MoeForCausalLM>},
      {"LlamaForCausalLM", "llama", &BuildFamily<llama::LlamaConfig, llama::LlamaForCausalLM>},
  });
  return *registry;
}

StatusOr<const ArchitectureEntry*> ArchitectureRegistry::Resolve(
    const std::string& model_type, const std::string& architecture) const {
  for (const ArchitectureEntry& entry : entries_) {
    if (!architecture.empty() && architecture == entry.architecture) return &entry;
  }
  for (const ArchitectureEntry& entry : entries_) {
    if (!model_type.empty() && model_type == entry.model_type) return &entry;
  }
  return NotFoundError("no registered architecture matches model_type '", model_type,
                      "' or architecture '", architecture, "'");
}

StatusOr<DataType> ResolveDType(DType requested, const HfConfig& config) {
  switch (requested) {
    case DType::kAuto:
    case DType::kBFloat16:
      // The kernels execute bfloat16; whatever the checkpoint stores
      // converts during the weight load.
      return DataType::kBFloat16;
    case DType::kFloat16:
    case DType::kFloat32:
      return UnimplementedError(
          "requested dtype is not supported; execution is bfloat16 (checkpoint stores ",
          config.Root().Has("torch_dtype") ? "torch_dtype" : "its own dtype", ")");
  }
  return InternalError("unreachable dtype request");
}

Status ModelLoadOptions::Validate() const {
  if (directory.empty()) {
    return InvalidArgumentError("ModelLoadOptions needs a checkpoint directory");
  }
  if (!device.IsCuda()) {
    return InvalidArgumentError("ModelLoadOptions needs a CUDA device, got ",
                                device.ToString());
  }
  return parallel.Validate();
}

StatusOr<DType> ModelLoadOptions::ParseDType(const std::string& text) {
  if (text.empty() || text == "auto") return DType::kAuto;
  if (text == "bfloat16" || text == "bf16") return DType::kBFloat16;
  if (text == "float16" || text == "half") return DType::kFloat16;
  if (text == "float32") return DType::kFloat32;
  return InvalidArgumentError("unknown dtype '", text,
                              "' (want auto, bfloat16, float16, or float32)");
}

StatusOr<std::unique_ptr<Model>> BuildModel(const ModelLoadOptions& options) {
  INFERX_RETURN_IF_ERROR(options.Validate());
  INFERX_ASSIGN_OR_RETURN(HfConfig config,
                          HfConfig::FromFile(options.directory + "/config.json"));
  INFERX_ASSIGN_OR_RETURN(
      const ArchitectureEntry* entry,
      ArchitectureRegistry::Builtins().Resolve(config.ModelType(), config.Architecture()));

  if (options.quantization == QuantizationPolicy::kAuto &&
      config.Root().Has("quantization_config")) {
    return UnimplementedError(
        "checkpoint declares quantization_config; dequantized loading is not "
        "restored in the rebuilt model layer yet");
  }

  ModelCreateParams params{config, options.device, options.parallel};
  return entry->create(params);
}

StatusOr<std::unique_ptr<Model>> LoadModel(const ModelLoadOptions& options) {
  INFERX_ASSIGN_OR_RETURN(std::unique_ptr<Model> model, BuildModel(options));

  if (options.policy.dummy_weights) {
    INFERX_RETURN_IF_ERROR(FillModelWeightsWithZeros(*model));
    return model;
  }

  INFERX_ASSIGN_OR_RETURN(CheckpointReader reader, CheckpointReader::Open(options.directory));
  if (!options.policy.strict_matching) {
    // Unexpected checkpoint tensors are skips, not errors, this once.
    auto* bindable = dynamic_cast<WeightBindable*>(model.get());
    if (bindable == nullptr) {
      return InternalError("model does not expose checkpoint bindings");
    }
    WeightMap weights;
    INFERX_RETURN_IF_ERROR(bindable->BuildWeightMap(weights));
    for (const std::string& name : reader.Names()) {
      if (weights.Find(name) == nullptr) weights.Skip(name);
    }
    INFERX_RETURN_IF_ERROR(LoadWeightMapInto(*model, weights, reader));
  } else {
    INFERX_RETURN_IF_ERROR(model->LoadWeights(reader));
  }

  if (options.policy.report_progress) {
    const ModelSpec& spec = model->GetModelSpec();
    INFERX_LOG(INFO) << "loaded " << spec.architecture << " from " << options.directory
                     << " (dtype " << DataTypeName(spec.dtype) << ", "
                     << spec.num_layers << " layers, vocab " << spec.vocab_size << ")";
  }
  return model;
}

}  // namespace models
}  // namespace inferx
