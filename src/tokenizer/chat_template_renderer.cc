#include "chat_template_renderer.h"

#include <chrono>
#include <utility>

namespace inferx {

StatusOr<ChatTemplateRenderer> ChatTemplateRenderer::Create(std::string source,
                                                            std::string bos_token,
                                                            std::string eos_token) {
  // minja::chat_template parses the source and probes it with dummy
  // messages to detect tool/system-role capabilities; both surface as
  // std::runtime_error on bad templates.
  std::unique_ptr<minja::chat_template> parsed;
  try {
    parsed = std::make_unique<minja::chat_template>(source, bos_token, eos_token);
  } catch (const std::exception& e) {
    return InvalidArgumentError("failed to parse chat template: ", e.what());
  }
  return ChatTemplateRenderer(std::move(source), std::move(bos_token), std::move(eos_token),
                              std::move(parsed));
}

ChatTemplateRenderer::ChatTemplateRenderer(std::string source, std::string bos_token,
                                           std::string eos_token,
                                           std::unique_ptr<minja::chat_template> template_impl)
    : source_(std::move(source)),
      bos_token_(std::move(bos_token)),
      eos_token_(std::move(eos_token)),
      template_(std::move(template_impl)) {}

StatusOr<std::string> ChatTemplateRenderer::Render(
    const nlohmann::ordered_json& messages, const nlohmann::ordered_json& tools,
    bool add_generation_prompt, const nlohmann::ordered_json& extra_context) const {
  minja::chat_template_inputs inputs;
  inputs.messages = messages;
  if (!tools.is_null()) inputs.tools = tools;
  inputs.add_generation_prompt = add_generation_prompt;
  if (!extra_context.is_null()) inputs.extra_context = extra_context;
  inputs.now = std::chrono::system_clock::now();

  minja::chat_template_options options;
  try {
    return template_->apply(inputs, options);
  } catch (const std::exception& e) {
    return InvalidArgumentError("chat template rendering failed: ", e.what());
  }
}

}  // namespace inferx
