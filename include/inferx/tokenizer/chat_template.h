#pragma once

#include <string>
#include <vector>

#include "inferx/core/status.h"

namespace inferx::tokenizer {

/// \brief One turn of a conversation.
struct ChatMessage {
  std::string role;     ///< "system", "user" or "assistant".
  std::string content;
};

/// \brief Renders `messages` into Qwen2's ChatML prompt format.
///
/// The checkpoint ships its chat template as Jinja, and the honest way to
/// render it is a Jinja engine. This is not that: it is a direct transcription
/// of the branch of Qwen2.5-Instruct's template that handles ordinary
/// conversations, which is the branch a chat completion takes.
///
/// The consequence is that it is right for this family of models and silently
/// wrong for any other, so it is named for the model rather than presented as a
/// general renderer. Serving a second architecture means either transcribing
/// its template here too, under its own name, or taking on a Jinja dependency
/// -- and by then the dependency is probably the better trade.
///
/// Not transcribed, because the API does not expose them yet: the tool-calling
/// branches. A message whose role is "tool" is rejected rather than dropped, so
/// that a request asking for something we do not implement fails visibly
/// instead of quietly losing a turn from the conversation.
///
/// \param messages              The conversation so far, in order.
/// \param add_generation_prompt Append the assistant header, which is what
///                              makes the model continue as the assistant
///                              rather than predicting the next speaker.
/// \return                      The prompt text, or InvalidArgument.
StatusOr<std::string> ApplyQwen2ChatTemplate(
    const std::vector<ChatMessage>& messages, bool add_generation_prompt);

}  // namespace inferx::tokenizer
