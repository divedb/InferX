#include <fstream>
#include <nlohmann/json.hpp>
#include <gtest/gtest.h>
#include "inferx/server/api.h"
#include "inferx/server/chat_template.h"

namespace inferx {
TEST(ChatTemplateTest, MatchesHuggingFaceReference) {
  std::ifstream input("tests/testdata/qwen3_chat_reference.json");
  ASSERT_TRUE(input.good());
  auto cases = Json::parse(input);
  for (const auto& fixture : cases) {
    Json request = {{"messages", fixture["messages"]},
                    {"chat_template_kwargs", {{"enable_thinking", fixture["enable_thinking"]}}}};
    SCOPED_TRACE(request.dump());
    auto parsed = ParseChatRequest(request.dump(), {}, "m");
    ASSERT_TRUE(parsed.ok) << parsed.message;
    auto rendered = RenderQwen3Chat(*parsed.request.chat);
    ASSERT_TRUE(rendered.ok()) << rendered.status();
    EXPECT_EQ(*rendered, fixture["rendered"].get<std::string>());
  }
}
TEST(ChatTemplateTest, RecognizesOnlyTheValidatedTemplate) {
  std::ifstream input("tests/testdata/qwen3_chat/tokenizer_config.json");
  const auto source = Json::parse(input)["chat_template"].get<std::string>();
  EXPECT_TRUE(IsSupportedChatTemplate(source));
  EXPECT_FALSE(IsSupportedChatTemplate(source + "modified"));
  EXPECT_FALSE(IsSupportedChatTemplate(""));
}
}  // namespace inferx
