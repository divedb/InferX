#ifndef INFERX_SERVER_CHAT_TEMPLATE_H_
#define INFERX_SERVER_CHAT_TEMPLATE_H_
#include <string>
#include <string_view>
#include "inferx/server/request_model.h"
namespace inferx {
bool IsSupportedChatTemplate(std::string_view source);
StatusOr<std::string> RenderQwen3Chat(const ChatPrompt& chat);
}  // namespace inferx
#endif
