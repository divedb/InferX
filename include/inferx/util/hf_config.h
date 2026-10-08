#pragma once

#include <memory>

#include "inferx/util/config_view.h"

namespace inferx {

/// \brief A parsed HuggingFace `config.json`.
///
/// Carries no model dimensions. It exists only between reading the file and running an
/// architecture parser: it supplies the dispatch key (`architecture()`) and checked access to
/// the JSON. Nothing after the parser holds it; the engine sees ModelSpec.
class HfConfig {
 public:
  static StatusOr<HfConfig> FromFile(const std::string& path);
  static StatusOr<HfConfig> FromJson(const std::string& json_text);

  /// \brief `architectures[0]`, e.g. "DeepseekV3ForCausalLM".
  ///
  /// \return The first string in the `architectures` array, or an empty string if the array is
  ///         missing or empty.
  const std::string& Architecture() const { return architecture_; }

  /// \brief The `model_type` string; empty if absent.
  ///
  /// \return The `model_type` string, or an empty string if the key is absent.
  const std::string& ModelType() const { return model_type_; }

  /// \brief The top-level object (torch_dtype, quantization_config, ...).
  ///
  /// \return The top-level object.
  const ConfigView& Root() const { return root_; }

  /// \brief The text model object: `text_config` for multimodal
  ///        checkpoints, otherwise the same object as Root().
  ///
  /// \return The text model configuration view.
  const ConfigView& Text() const { return text_; }

 private:
  std::shared_ptr<const nlohmann::json> doc_;  // views point into this
  std::string architecture_;
  std::string model_type_;
  ConfigView root_;
  ConfigView text_;
};

}  // namespace inferx
