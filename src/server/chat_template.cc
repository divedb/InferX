#include "inferx/server/chat_template.h"

namespace inferx {
namespace {
constexpr std::string_view kQwen3Template =
#include "qwen3_chat_template.inc"
;
std::string TrimNewlines(std::string text, bool left = true, bool right = true) {
  if (left) text.erase(0, text.find_first_not_of('\n'));
  if (right) text.erase(text.find_last_not_of('\n') + 1);
  return text;
}
}
bool IsSupportedChatTemplate(std::string_view source) { return source == kQwen3Template; }

StatusOr<std::string> RenderQwen3Chat(const ChatPrompt& chat) {
  if (chat.messages.empty()) return InvalidArgumentError("messages must not be empty");
  if (chat.options.continue_final_message) return UnimplementedError("continue_final_message is unsupported");
  std::size_t last_query = chat.messages.size() - 1;
  for (std::size_t i = 0; i < chat.messages.size(); ++i) {
    if (chat.messages[i].role == Role::kUser) {
      std::string content;
      for (const auto& part : chat.messages[i].content)
        if (const auto* text = std::get_if<TextPart>(&part)) content += text->text;
      if (!(content.starts_with("<tool_response>") && content.ends_with("</tool_response>")))
        last_query = i;
    }
  }
  std::string output;
  for (std::size_t i = 0; i < chat.messages.size(); ++i) {
    const auto& message = chat.messages[i];
    if (message.role == Role::kTool || !message.name.empty())
      return UnimplementedError("tool and named messages are unsupported");
    std::string content;
    for (const auto& part : message.content) {
      const auto* text = std::get_if<TextPart>(&part);
      if (!text) return UnimplementedError("multimodal chat is unsupported");
      content += text->text;
    }
    const char* role = message.role == Role::kSystem ? "system" :
                       message.role == Role::kUser ? "user" : "assistant";
    if (message.role == Role::kAssistant) {
      std::string reasoning;
      const auto first_end = content.find("</think>");
      if (first_end != std::string::npos) {
        reasoning = TrimNewlines(content.substr(0, first_end), false, true);
        const auto start = reasoning.rfind("<think>");
        if (start != std::string::npos) reasoning.erase(0, start + 7);
        reasoning = TrimNewlines(reasoning, true, false);
        content = TrimNewlines(content.substr(content.rfind("</think>") + 8), true, false);
      }
      if (i > last_query && (i + 1 == chat.messages.size() || !reasoning.empty())) {
        content = "<think>\n" + TrimNewlines(reasoning) + "\n</think>\n\n" +
                  TrimNewlines(content, true, false);
      }
    }
    output += std::string("<|im_start|>") + role + "\n" + content + "<|im_end|>\n";
  }
  if (chat.options.add_generation_prompt) {
    output += "<|im_start|>assistant\n";
    if (!chat.options.enable_thinking) output += "<think>\n\n</think>\n\n";
  }
  return output;
}
}  // namespace inferx
