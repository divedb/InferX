#include "inferx/util/hf_config.h"

#include <fstream>
#include <nlohmann/json.hpp>
#include <sstream>

namespace inferx {

StatusOr<HfConfig> HfConfig::FromFile(const std::string& path) {
  std::ifstream in(path, std::ios::binary);

  if (!in) return NotFoundError("cannot open ", path);

  std::ostringstream buffer;
  buffer << in.rdbuf();
  auto cfg = FromJson(buffer.str());

  if (!cfg.ok()) {
    return absl::Status(cfg.status().code(), absl::StrCat(path, ": ", cfg.status().message()));
  }

  return cfg;
}

StatusOr<HfConfig> HfConfig::FromJson(const std::string& json_text) {
  nlohmann::json parsed = nlohmann::json::parse(json_text, nullptr, /*allow_exceptions=*/false);

  if (parsed.is_discarded()) {
    return InvalidArgumentError("config is not valid JSON");
  }

  if (!parsed.is_object()) {
    return InvalidArgumentError("config root must be an object");
  }

  HfConfig cfg;
  cfg.doc_ = std::make_shared<const nlohmann::json>(std::move(parsed));

  const auto archs = cfg.doc_->find("architectures");

  if (archs == cfg.doc_->end() || !archs->is_array() || archs->empty() ||
      !(*archs)[0].is_string()) {
    return InvalidArgumentError("config key 'architectures' must be a non-empty string list");
  }

  cfg.architecture_ = (*archs)[0].get<std::string>();
  const auto model_type = cfg.doc_->find("model_type");

  if (model_type != cfg.doc_->end() && model_type->is_string()) {
    cfg.model_type_ = model_type->get<std::string>();
  }

  cfg.root_ = ConfigView(cfg.doc_.get(), "");

  // Multimodal wrappers keep the text model under text_config; text-only
  // checkpoints parse their root directly.
  const auto text = cfg.doc_->find("text_config");

  if (text != cfg.doc_->end() && text->is_object()) {
    cfg.text_ = ConfigView(&*text, "text_config");
  } else {
    cfg.text_ = cfg.root_;
  }
  return cfg;
}

}  // namespace inferx