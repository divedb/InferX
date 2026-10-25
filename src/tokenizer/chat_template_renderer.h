/// \brief Chat-template rendering over the vendored minja subset.
///
/// Wraps minja::chat_template (the same engine llama.cpp uses) behind
/// Status-returning calls: template parse/probe failures surface at
/// construction, render failures per call. Rendering applies the
/// transformers-compatibility polyfills (tools, tool calls, system role,
/// typed content) and exposes bos/eos tokens, tools, and extra template
/// variables to the template, matching vLLM's apply_hf_chat_template
/// inputs. apply() is const and the parsed AST is immutable after
/// construction, so concurrent Render calls on one renderer are safe —
/// but see HFTokenizer for the instance-level threading contract.
#ifndef INFERX_TOKENIZER_CHAT_TEMPLATE_RENDERER_H_
#define INFERX_TOKENIZER_CHAT_TEMPLATE_RENDERER_H_

#include <memory>
#include <string>
#include <utility>

#include <nlohmann/json.hpp>
#include <minja/chat-template.hpp>

#include "inferx/core/status.h"

namespace inferx {

class ChatTemplateRenderer {
 public:
  /// \brief Parses `source` as a Jinja chat template.
  ///
  /// \param source    Template text (tokenizer_config chat_template or a
  ///                  per-request override).
  /// \param bos_token Bos token string, or "" when the tokenizer has none.
  /// \param eos_token Eos token string, or "" when the tokenizer has none.
  static StatusOr<ChatTemplateRenderer> Create(std::string source, std::string bos_token,
                                               std::string eos_token);

  ChatTemplateRenderer(ChatTemplateRenderer&&) = default;
  ChatTemplateRenderer& operator=(ChatTemplateRenderer&&) = default;

  /// \brief Renders messages through the template.
  ///
  /// \param messages             JSON array of {role, content[, name,
  ///                             tool_calls]} message objects.
  /// \param tools                JSON array of tool definitions, or null to
  ///                             leave the `tools` variable undefined.
  /// \param add_generation_prompt Passes through to the template.
  /// \param extra_context        Additional template variables
  ///                             (vLLM chat_template_kwargs); must be an
  ///                             object or null.
  /// \return                     The rendered prompt, or InvalidArgument on
  ///                             a rendering error.
  StatusOr<std::string> Render(const nlohmann::ordered_json& messages,
                               const nlohmann::ordered_json& tools, bool add_generation_prompt,
                               const nlohmann::ordered_json& extra_context) const;

  /// \brief The template source this renderer was built from.
  const std::string& source() const noexcept { return source_; }

 private:
  ChatTemplateRenderer(std::string source, std::string bos_token, std::string eos_token,
                       std::unique_ptr<minja::chat_template> template_impl);

  std::string source_;
  std::string bos_token_;
  std::string eos_token_;
  std::unique_ptr<minja::chat_template> template_;
};

}  // namespace inferx

#endif  // INFERX_TOKENIZER_CHAT_TEMPLATE_RENDERER_H_
