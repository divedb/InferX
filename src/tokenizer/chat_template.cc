#include "inferx/tokenizer/chat_template.h"

namespace inferx::tokenizer {
namespace {

// The template supplies this when the caller gives no system message. Copied
// exactly, including the wording -- the model was tuned with it, and a
// paraphrase is a different prompt.
constexpr std::string_view kDefaultSystemPrompt =
    "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.";

void AppendTurn(std::string_view role, std::string_view content,
                std::string* out) {
  out->append("<|im_start|>");
  out->append(role);
  out->push_back('\n');
  out->append(content);
  out->append("<|im_end|>\n");
}

}  // namespace

StatusOr<std::string> ApplyQwen2ChatTemplate(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt) {
  for (const ChatMessage& message : messages) {
    if (message.role != "system" && message.role != "user" &&
        message.role != "assistant") {
      return InvalidArgumentError(
          "unsupported message role \"", message.role,
          "\"; this template renders system, user and assistant turns only");
    }
  }

  std::string out;

  // The system turn is emitted ahead of the loop, present or not, because the
  // template guarantees the prompt opens with one.
  if (!messages.empty() && messages.front().role == "system") {
    AppendTurn("system", messages.front().content, &out);
  } else {
    AppendTurn("system", kDefaultSystemPrompt, &out);
  }

  for (size_t i = 0; i < messages.size(); ++i) {
    // Skipped rather than repeated: a leading system message has already been
    // emitted above. One appearing later in the conversation is a real turn.
    if (i == 0 && messages[i].role == "system") continue;

    AppendTurn(messages[i].role, messages[i].content, &out);
  }

  if (add_generation_prompt) out.append("<|im_start|>assistant\n");

  return out;
}

}  // namespace inferx::tokenizer
