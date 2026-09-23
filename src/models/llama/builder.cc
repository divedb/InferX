#include "inferx/models/llama/builder.h"

#include "inferx/models/lm/loader.h"
#include "models/lm/config_parser.h"

namespace inferx::llama {

StatusOr<lm::DecoderConfig> ParseConfig(const std::string& text) {
  try {
    const auto j = nlohmann::json::parse(text);
    if (j.value("model_type", std::string()) != "llama") {
      return InvalidArgumentError("expected model_type llama");
    }
    INFERX_ASSIGN_OR_RETURN(auto config, lm::AttentionDecoderConfig(j, /*qk_norm=*/false,
                                                                    /*plus_one_norm=*/false));
    INFERX_RETURN_IF_ERROR(config.Validate());
    return config;
  } catch (const nlohmann::json::exception& e) {
    return InvalidArgumentError("invalid Llama config: ", e.what());
  }
}

StatusOr<std::unique_ptr<Model>> Load(const std::string& directory, DeviceId device,
                                      int max_tokens, int max_seqs, ops::AttentionBackend backend) {
  INFERX_ASSIGN_OR_RETURN(auto text, lm::ReadConfig(directory));
  INFERX_ASSIGN_OR_RETURN(auto config, ParseConfig(text));
  config.attention_backend = backend;
  return lm::LoadCausalLM(directory, config, {}, device, max_tokens, max_seqs);
}

}  // namespace inferx::llama
