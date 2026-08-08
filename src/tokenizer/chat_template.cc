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

StatusOr<std::string> ApplyDeepSeekV2ChatTemplate(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt) {
  for (const ChatMessage& message : messages) {
    if (message.role != "system" && message.role != "user" &&
        message.role != "assistant") {
      return InvalidArgumentError(
          "unsupported message role \"", message.role,
          "\"; this template renders system, user and assistant turns only");
    }
  }

  // The BOS sentinel, as text: chat tokenization runs with specials-as-control,
  // which folds it to DeepSeek's bos_token_id.
  std::string out = "<｜begin▁of▁sentence｜>";

  // A direct transcription of the template's loop: system messages are bare
  // text, wherever they appear, and there is no default system prompt.
  for (const ChatMessage& message : messages) {
    if (message.role == "system") {
      out.append(message.content);
      out.append("\n\n");
    } else if (message.role == "user") {
      out.append("User: ");
      out.append(message.content);
      out.append("\n\n");
    } else {
      out.append("Assistant: ");
      out.append(message.content);
      out.append("<｜end▁of▁sentence｜>");
    }
  }

  // No trailing space: the template emits exactly `Assistant:` and the model
  // was tuned to produce the space itself.
  if (add_generation_prompt) out.append("Assistant:");

  return out;
}

StatusOr<std::string> ApplyChatTemplate(ChatTemplateKind kind,
                                        const std::vector<ChatMessage>& messages,
                                        bool add_generation_prompt) {
  switch (kind) {
    case ChatTemplateKind::kQwen2:
      return ApplyQwen2ChatTemplate(messages, add_generation_prompt);
    case ChatTemplateKind::kDeepSeekV2:
      return ApplyDeepSeekV2ChatTemplate(messages, add_generation_prompt);
  }
  return InvalidArgumentError("unknown chat template kind");
}

}  // namespace inferx::tokenizer
