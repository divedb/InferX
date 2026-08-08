/// The transcribed chat templates, against strings derived from running each
/// checkpoint's own Jinja template through HF's apply_chat_template.
///
/// A template error is the quietest failure in serving: the model still
/// answers, just worse, and nothing downstream can tell. These fixtures are
/// therefore exact string comparisons, not substring checks -- a missing
/// newline or an extra space after `Assistant:` is precisely the kind of bug
/// they exist to catch.

#include "inferx/tokenizer/chat_template.h"

#include <string>
#include <vector>

#include <gtest/gtest.h>

namespace inferx::tokenizer {
namespace {

TEST(DeepSeekV2ChatTemplate, RendersSystemUserAssistantTurns) {
  const std::vector<ChatMessage> messages{
      {"system", "Answer briefly."},
      {"user", "What is MLA?"},
      {"assistant", "Compressed-latent attention."},
      {"user", "Shorter."},
  };

  auto got = ApplyDeepSeekV2ChatTemplate(messages,
                                         /*add_generation_prompt=*/true);
  ASSERT_TRUE(got.ok()) << got.status();

  // From DeepSeek-V2-Lite-Chat's tokenizer_config.json template: BOS, bare
  // system text, User/Assistant prefixes, assistant turns closed by the EOS
  // sentinel, and a generation prompt with no trailing space.
  EXPECT_EQ(*got,
            "<｜begin▁of▁sentence｜>"
            "Answer briefly.\n\n"
            "User: What is MLA?\n\n"
            "Assistant: Compressed-latent attention.<｜end▁of▁sentence｜>"
            "User: Shorter.\n\n"
            "Assistant:");
}

TEST(DeepSeekV2ChatTemplate, InjectsNoDefaultSystemPrompt) {
  // Unlike Qwen2's template, a conversation without a system message starts
  // straight at the first turn -- injecting one would be a different prompt
  // than the model was tuned for.
  const std::vector<ChatMessage> messages{{"user", "Hi"}};

  auto got = ApplyDeepSeekV2ChatTemplate(messages,
                                         /*add_generation_prompt=*/true);
  ASSERT_TRUE(got.ok()) << got.status();

  EXPECT_EQ(*got,
            "<｜begin▁of▁sentence｜>"
            "User: Hi\n\n"
            "Assistant:");
}

TEST(DeepSeekV2ChatTemplate, OmitsTheGenerationPromptWhenNotAsked) {
  const std::vector<ChatMessage> messages{
      {"user", "Hi"},
      {"assistant", "Hello."},
  };

  auto got = ApplyDeepSeekV2ChatTemplate(messages,
                                         /*add_generation_prompt=*/false);
  ASSERT_TRUE(got.ok()) << got.status();

  EXPECT_EQ(*got,
            "<｜begin▁of▁sentence｜>"
            "User: Hi\n\n"
            "Assistant: Hello.<｜end▁of▁sentence｜>");
}

TEST(DeepSeekV2ChatTemplate, RejectsUnsupportedRoles) {
  const std::vector<ChatMessage> messages{{"tool", "{}"}};

  const auto got = ApplyDeepSeekV2ChatTemplate(messages, true);
  ASSERT_FALSE(got.ok());
  EXPECT_EQ(got.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(ChatTemplateKind, DispatchesToTheNamedTemplate) {
  const std::vector<ChatMessage> messages{{"user", "Hi"}};

  auto qwen = ApplyChatTemplate(ChatTemplateKind::kQwen2, messages, true);
  ASSERT_TRUE(qwen.ok()) << qwen.status();
  EXPECT_NE(qwen->find("<|im_start|>"), std::string::npos);

  auto deepseek =
      ApplyChatTemplate(ChatTemplateKind::kDeepSeekV2, messages, true);
  ASSERT_TRUE(deepseek.ok()) << deepseek.status();
  EXPECT_EQ(deepseek->find("<|im_start|>"), std::string::npos);
  EXPECT_NE(deepseek->find("<｜begin▁of▁sentence｜>"), std::string::npos);

  // The two families' prompts must never coincide: identical output would
  // mean the dispatch is not actually selecting.
  EXPECT_NE(*qwen, *deepseek);
}

}  // namespace
}  // namespace inferx::tokenizer
