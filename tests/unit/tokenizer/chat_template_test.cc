// Chat-template application tests over the qwen3_chat fixture: parity with
// the HuggingFace-rendered goldens in qwen3_chat_reference.json (including
// the enable_thinking template kwarg), generation-prompt and
// continue_final_message semantics (transformers behavior), per-request
// template overrides, and error paths.
#include <fstream>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include "inferx/core/status.h"
#include "inferx/tokenizer/hf_tokenizer.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx {
namespace {

constexpr const char* kQwen3Dir = "tests/testdata/qwen3_chat";
constexpr const char* kReferencePath = "tests/testdata/qwen3_chat_reference.json";

class HFTokenizerChatTemplateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto loaded = HFTokenizer::Load(kQwen3Dir);
    ASSERT_TRUE(loaded.ok()) << loaded.status();
    tokenizer_ = std::move(*loaded);
  }

  StatusOr<std::string> Apply(const std::string& messages_json,
                              const ChatTemplateOptions& options = {}) const {
    return tokenizer_->ApplyChatTemplate(messages_json, options);
  }

  std::shared_ptr<HFTokenizer> tokenizer_;
};

// The goldens are transformers apply_chat_template outputs captured for the
// fixture's template: messages plus the enable_thinking kwarg.
TEST_F(HFTokenizerChatTemplateTest, MatchesHfRenderedGoldens) {
  std::ifstream input(kReferencePath);
  ASSERT_TRUE(input.good()) << "missing golden file " << kReferencePath;
  const nlohmann::json cases = nlohmann::json::parse(input);
  ASSERT_GT(cases.size(), 0u);

  for (const auto& golden : cases) {
    ChatTemplateOptions options;
    options.template_kwargs_json =
        nlohmann::json({{"enable_thinking", golden["enable_thinking"]}}).dump();
    const auto rendered = Apply(golden["messages"].dump(), options);
    ASSERT_TRUE(rendered.ok()) << rendered.status() << "\nmessages: "
                               << golden["messages"].dump();
    EXPECT_EQ(*rendered, golden["rendered"].get<std::string>())
        << "messages: " << golden["messages"].dump();
  }
}

TEST_F(HFTokenizerChatTemplateTest, TypedContentPartsCollapseToText) {
  const auto typed = Apply(R"([{"role":"user","content":[{"type":"text","text":"hi"}]}])");
  ASSERT_TRUE(typed.ok()) << typed.status();
  const auto plain = Apply(R"([{"role":"user","content":"hi"}])");
  ASSERT_TRUE(plain.ok()) << plain.status();
  EXPECT_EQ(*typed, *plain);
}

TEST_F(HFTokenizerChatTemplateTest, NoGenerationPromptDropsTheAssistantHeader) {
  const auto with = Apply(R"([{"role":"user","content":"hi"}])");
  ASSERT_TRUE(with.ok()) << with.status();
  ChatTemplateOptions without;
  without.add_generation_prompt = false;
  const auto result = Apply(R"([{"role":"user","content":"hi"}])", without);
  ASSERT_TRUE(result.ok()) << result.status();
  EXPECT_NE(*result, *with);
  EXPECT_EQ(result->find("<|im_start|>assistant"), std::string::npos)
      << "no assistant header expected: " << *result;
}

TEST_F(HFTokenizerChatTemplateTest, ContinueFinalMessageCutsAtPrefilledText) {
  // transformers semantics: render with the final assistant message, then
  // cut right after that message's content; add_generation_prompt must be
  // off (either explicitly or implied by continue_final_message).
  ChatTemplateOptions options;
  options.add_generation_prompt = false;
  options.continue_final_message = true;
  const auto rendered = Apply(
      R"([{"role":"user","content":"hi"},{"role":"assistant","content":"2 + 2 = "}])", options);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  // The template itself appends an empty <think> block to assistant
  // history messages; the cut happens at the sentinel tag position, so the
  // pre-filled content's trailing spacing survives (transformers >= 5
  // behavior, verified against the live library).
  EXPECT_EQ(*rendered, "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
                       "<think>\n\n</think>\n\n2 + 2 = ")
      << "got: " << *rendered;
}

TEST_F(HFTokenizerChatTemplateTest, ContinueFinalMessageExcludesGenerationPrompt) {
  ChatTemplateOptions options;
  options.add_generation_prompt = true;
  options.continue_final_message = true;
  const auto rendered = Apply(R"([{"role":"user","content":"hi"}])", options);
  ASSERT_FALSE(rendered.ok());
  EXPECT_EQ(rendered.status().code(), absl::StatusCode::kInvalidArgument);

  // A final message without content cannot be continued.
  ChatTemplateOptions no_content;
  no_content.add_generation_prompt = false;
  no_content.continue_final_message = true;
  const auto missing = Apply(
      R"([{"role":"user","content":"hi"},{"role":"assistant","content":null}])", no_content);
  ASSERT_FALSE(missing.ok());
  EXPECT_EQ(missing.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(HFTokenizerChatTemplateTest, PerRequestTemplateOverrideWins) {
  ChatTemplateOptions options;
  options.chat_template = "{% for message in messages %}{{ message['role'] | upper }}:"
                          "{{ message['content'] }};{% endfor %}";
  const auto rendered = Apply(R"([{"role":"user","content":"hi"}])", options);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_EQ(*rendered, "USER:hi;");

  // A broken override fails without poisoning the checkpoint template.
  ChatTemplateOptions broken;
  broken.chat_template = "{% for message in messages %}";  // unclosed tag
  const auto failed = Apply(R"([{"role":"user","content":"hi"}])", broken);
  ASSERT_FALSE(failed.ok());
  const auto recovered = Apply(R"([{"role":"user","content":"hi"}])");
  ASSERT_TRUE(recovered.ok()) << recovered.status();
}

TEST_F(HFTokenizerChatTemplateTest, ToolsReachTheTemplate) {
  // The Qwen3 template renders tool definitions when `tools` is set.
  ChatTemplateOptions options;
  options.tools_json = R"([
    {"type": "function", "function": {
      "name": "get_weather", "description": "Get the weather",
      "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
    }}
  ])";
  const auto rendered = Apply(R"([{"role":"user","content":"weather in Paris?"}])", options);
  ASSERT_TRUE(rendered.ok()) << rendered.status();
  EXPECT_NE(rendered->find("get_weather"), std::string::npos) << *rendered;

  // Without tools the tool section is absent.
  const auto plain = Apply(R"([{"role":"user","content":"weather in Paris?"}])");
  ASSERT_TRUE(plain.ok()) << plain.status();
  EXPECT_EQ(plain->find("get_weather"), std::string::npos);
}

TEST_F(HFTokenizerChatTemplateTest, BadInputsAreInvalidArgument) {
  for (const char* bad : {"", "   ", "{ not json", R"({"role":"user"})", R"("string")",
                          R"([{"content":"no role"}])", R"([{"role": 5, "content":"x"}])"}) {
    const auto rendered = Apply(bad);
    EXPECT_FALSE(rendered.ok()) << "input: " << bad;
    EXPECT_EQ(rendered.status().code(), absl::StatusCode::kInvalidArgument)
        << "input: " << bad << " status: " << rendered.status();
  }

  // Non-text content parts are explicitly rejected while multimodal input
  // is out of scope.
  const auto image = Apply(R"([{"role":"user","content":[{"type":"image_url"}]}])");
  ASSERT_FALSE(image.ok());
  EXPECT_EQ(image.status().code(), absl::StatusCode::kUnimplemented);

  ChatTemplateOptions bad_tools;
  bad_tools.tools_json = R"({"not": "an array"})";
  const auto tools = Apply(R"([{"role":"user","content":"hi"}])", bad_tools);
  ASSERT_FALSE(tools.ok());
  EXPECT_EQ(tools.status().code(), absl::StatusCode::kInvalidArgument);

  ChatTemplateOptions bad_kwargs;
  bad_kwargs.template_kwargs_json = R"(["not", "an", "object"])";
  const auto kwargs = Apply(R"([{"role":"user","content":"hi"}])", bad_kwargs);
  ASSERT_FALSE(kwargs.ok());
  EXPECT_EQ(kwargs.status().code(), absl::StatusCode::kInvalidArgument);
}

TEST(HFTokenizerChatTemplateNoFixtureTest, MissingTemplateIsExplicit) {
  auto tokenizer = HFTokenizer::Load("tests/testdata/tiny_tokenizer");
  ASSERT_TRUE(tokenizer.ok()) << tokenizer.status();
  const auto rendered = (*tokenizer)->ApplyChatTemplate(R"([{"role":"user","content":"hi"}])");
  ASSERT_FALSE(rendered.ok());
  EXPECT_EQ(rendered.status().code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace inferx
