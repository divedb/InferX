#include "inferx/api/openai.h"

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace inferx::api {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;

constexpr std::string_view kChatPrefix =
    R"({"model":"m","messages":[{"role":"user","content":"hi"}])";

std::string Chat(std::string_view extra) {
  std::string body(kChatPrefix);
  if (!extra.empty()) {
    body += ",";
    body += extra;
  }
  body += "}";
  return body;
}

std::string Completion(std::string_view extra) {
  std::string body = R"({"model":"m","prompt":"hi")";
  if (!extra.empty()) {
    body += ",";
    body += extra;
  }
  body += "}";
  return body;
}

TEST(ChatRequestTest, DefaultsMatchDocumentedBehavior) {
  auto request = ParseChatCompletionRequest(Chat(""));
  ASSERT_TRUE(request.ok());
  EXPECT_EQ(request->sampling.max_tokens, 128);
  EXPECT_FLOAT_EQ(request->sampling.temperature, 0.0f);
  EXPECT_EQ(request->sampling.top_k, 0);
  EXPECT_FLOAT_EQ(request->sampling.min_p, 0.0f);
  EXPECT_EQ(request->sampling.n, 1);
  EXPECT_TRUE(request->sampling.skip_special_tokens);
  EXPECT_FALSE(request->sampling.ignore_eos);
  EXPECT_FALSE(request->sampling.want_logprobs);
}

TEST(ChatRequestTest, VllmSamplingExtensionsParse) {
  auto request = ParseChatCompletionRequest(Chat(
      R"("top_k":40,"min_p":0.05,"presence_penalty":0.5,)"
      R"("frequency_penalty":-0.5,"repetition_penalty":1.1,)"
      R"("stop_token_ids":[7,11],"ignore_eos":true,"min_tokens":4,)"
      R"("skip_special_tokens":false,"include_stop_str_in_output":true,)"
      R"("n":2,"seed":9)"));
  ASSERT_TRUE(request.ok());
  const SamplingRequest& sampling = request->sampling;
  EXPECT_EQ(sampling.top_k, 40);
  EXPECT_FLOAT_EQ(sampling.min_p, 0.05f);
  EXPECT_FLOAT_EQ(sampling.presence_penalty, 0.5f);
  EXPECT_FLOAT_EQ(sampling.frequency_penalty, -0.5f);
  EXPECT_FLOAT_EQ(sampling.repetition_penalty, 1.1f);
  EXPECT_THAT(sampling.stop_token_ids, ElementsAre(7, 11));
  EXPECT_TRUE(sampling.ignore_eos);
  EXPECT_EQ(sampling.min_tokens, 4);
  EXPECT_FALSE(sampling.skip_special_tokens);
  EXPECT_TRUE(sampling.include_stop_str_in_output);
  EXPECT_EQ(sampling.n, 2);
  EXPECT_TRUE(sampling.has_seed);
}

TEST(ChatRequestTest, MaxCompletionTokensAliasesMaxTokens) {
  auto request =
      ParseChatCompletionRequest(Chat(R"("max_completion_tokens":64)"));
  ASSERT_TRUE(request.ok());
  EXPECT_EQ(request->sampling.max_tokens, 64);

  auto agree = ParseChatCompletionRequest(
      Chat(R"("max_tokens":64,"max_completion_tokens":64)"));
  EXPECT_TRUE(agree.ok());

  auto disagree = ParseChatCompletionRequest(
      Chat(R"("max_tokens":64,"max_completion_tokens":65)"));
  ASSERT_FALSE(disagree.ok());
  EXPECT_THAT(std::string(disagree.status().message()),
              HasSubstr("disagree"));
}

TEST(ChatRequestTest, TopKMinusOneMeansDisabled) {
  auto request = ParseChatCompletionRequest(Chat(R"("top_k":-1)"));
  ASSERT_TRUE(request.ok());
  EXPECT_EQ(request->sampling.top_k, 0);
}

TEST(ChatRequestTest, UnsupportedFeaturesGetClear400s) {
  const struct {
    std::string_view extra;
    std::string_view names;
  } kCases[] = {
      {R"("logit_bias":{"50256":-100})", "logit biasing"},
      {R"("response_format":{"type":"json_object"})", "structured outputs"},
      {R"("guided_json":{"type":"object"})", "structured outputs"},
      {R"("use_beam_search":true)", "beam search"},
      {R"("tools":[{"type":"function"}])", "tool calling"},
      {R"("tool_choice":"auto")", "tool calling"},
      {R"("prompt_logprobs":1)", "prompt logprobs"},
      {R"("bad_words":["x"])", "bad-words"},
      {R"("allowed_token_ids":[1])", "allow-lists"},
      {R"("truncate_prompt_tokens":10)", "truncation"},
      {R"("spaces_between_special_tokens":false)", "not supported"},
      {R"("echo":true)", "/v1/completions"},
  };
  for (const auto& test_case : kCases) {
    auto request = ParseChatCompletionRequest(Chat(test_case.extra));
    ASSERT_FALSE(request.ok()) << test_case.extra;
    EXPECT_TRUE(absl::IsInvalidArgument(request.status())) << test_case.extra;
    EXPECT_THAT(std::string(request.status().message()),
                HasSubstr(std::string(test_case.names)))
        << test_case.extra;
  }
}

TEST(ChatRequestTest, BenignVariantsStillParse) {
  EXPECT_TRUE(ParseChatCompletionRequest(
                  Chat(R"("response_format":{"type":"text"})"))
                  .ok());
  EXPECT_TRUE(ParseChatCompletionRequest(Chat(R"("tool_choice":"none")")).ok());
  EXPECT_TRUE(ParseChatCompletionRequest(Chat(R"("tools":[])")).ok());
  EXPECT_TRUE(ParseChatCompletionRequest(Chat(R"("best_of":1)")).ok());
  // Unknown fields keep OpenAI's forward-compatible tolerance.
  EXPECT_TRUE(ParseChatCompletionRequest(
                  Chat(R"("user":"u1","store":false,"metadata":{})"))
                  .ok());
}

TEST(ChatRequestTest, ChatLogprobsSpelling) {
  auto request = ParseChatCompletionRequest(
      Chat(R"("logprobs":true,"top_logprobs":5)"));
  ASSERT_TRUE(request.ok());
  EXPECT_TRUE(request->sampling.want_logprobs);
  EXPECT_EQ(request->sampling.top_logprobs, 5);

  EXPECT_FALSE(
      ParseChatCompletionRequest(Chat(R"("top_logprobs":5)")).ok());
  EXPECT_FALSE(ParseChatCompletionRequest(
                   Chat(R"("logprobs":true,"top_logprobs":21)"))
                   .ok());
}

TEST(ChatRequestTest, MinTokensBoundedByMaxTokens) {
  EXPECT_TRUE(ParseChatCompletionRequest(
                  Chat(R"("max_tokens":8,"min_tokens":8)"))
                  .ok());
  EXPECT_FALSE(ParseChatCompletionRequest(
                   Chat(R"("max_tokens":8,"min_tokens":9)"))
                   .ok());
}

TEST(CompletionRequestTest, IntegerLogprobsAndEcho) {
  auto request =
      ParseCompletionRequest(Completion(R"("logprobs":3,"echo":true)"));
  ASSERT_TRUE(request.ok());
  EXPECT_TRUE(request->sampling.want_logprobs);
  EXPECT_EQ(request->sampling.top_logprobs, 3);
  EXPECT_TRUE(request->sampling.echo);

  EXPECT_TRUE(
      ParseCompletionRequest(Completion(R"("logprobs":null)")).ok());
  EXPECT_FALSE(
      ParseCompletionRequest(Completion(R"("logprobs":21)")).ok());
}

TEST(CompletionRequestTest, SuffixIsRejected) {
  auto request = ParseCompletionRequest(Completion(R"("suffix":"tail")"));
  ASSERT_FALSE(request.ok());
  EXPECT_THAT(std::string(request.status().message()),
              HasSubstr("fill-in-the-middle"));
}

TEST(CompletionRequestTest, NBounds) {
  EXPECT_TRUE(ParseCompletionRequest(Completion(R"("n":4)")).ok());
  EXPECT_FALSE(ParseCompletionRequest(Completion(R"("n":0)")).ok());
  EXPECT_FALSE(ParseCompletionRequest(Completion(R"("n":65)")).ok());
  EXPECT_FALSE(
      ParseCompletionRequest(Completion(R"("n":2,"best_of":4)")).ok());
}

// A single choice without logprobs must render the exact pre-logprobs bytes:
// transport tests and clients snapshot this shape.
TEST(ResponseJsonTest, SingleChoiceWithoutLogprobsKeepsLegacyShape) {
  std::vector<ChatChoice> chat(1);
  chat[0].content = "ok";
  EXPECT_EQ(
      ChatCompletionJson("req_1", "m", chat, {2, 1}, 9),
      "{\"id\":\"req_1\",\"object\":\"chat.completion\",\"created\":9,"
      "\"model\":\"m\",\"system_fingerprint\":\"greedy\",\"choices\":"
      "[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":\"ok\"},"
      "\"logprobs\":null,\"finish_reason\":\"stop\"}],\"usage\":"
      "{\"prompt_tokens\":2,\"completion_tokens\":1,\"total_tokens\":3}}");

  std::vector<CompletionChoice> text(1);
  text[0].text = "ok";
  EXPECT_EQ(
      CompletionJson("req_1", "m", text, {2, 1}, 9),
      "{\"id\":\"req_1\",\"object\":\"text_completion\",\"created\":9,"
      "\"model\":\"m\",\"system_fingerprint\":\"greedy\",\"choices\":"
      "[{\"index\":0,\"text\":\"ok\",\"logprobs\":null,"
      "\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":2,"
      "\"completion_tokens\":1,\"total_tokens\":3}}");
}

TEST(ResponseJsonTest, ChatLogprobsCarryTokenBytesAndAlternatives) {
  ChoiceLogprobs logprobs;
  TokenLogprob token;
  token.token = "hi";
  token.logprob = -0.5f;
  token.top = {{"hi", -0.5f}, {"yo", -2.0f}};
  logprobs.tokens.push_back(std::move(token));
  std::vector<ChatChoice> choices(1);
  choices[0].content = "hi";
  choices[0].logprobs = &logprobs;
  EXPECT_THAT(
      ChatCompletionJson("req_1", "m", choices, {3, 1}, 7),
      HasSubstr(
          "\"logprobs\":{\"content\":[{\"token\":\"hi\","
          "\"logprob\":-0.500000,\"bytes\":[104,105],\"top_logprobs\":"
          "[{\"token\":\"hi\",\"logprob\":-0.500000,\"bytes\":[104,105]},"
          "{\"token\":\"yo\",\"logprob\":-2.000000,\"bytes\":[121,111]}]}]}"));
}

// `top_logprobs: 0` still reports the chosen token; the alternatives array is
// present and empty, matching OpenAI's shape.
TEST(ResponseJsonTest, ChatLogprobsWithoutAlternativesEmitEmptyArray) {
  ChoiceLogprobs logprobs;
  logprobs.tokens.push_back({"a", -0.125f, {}});
  std::vector<ChatChoice> choices(1);
  choices[0].content = "a";
  choices[0].logprobs = &logprobs;
  EXPECT_THAT(ChatCompletionJson("req_1", "m", choices, {1, 1}, 7),
              HasSubstr("\"logprob\":-0.125000,\"bytes\":[97],"
                        "\"top_logprobs\":[]}"));
}

// A multibyte token's `bytes` are its raw UTF-8 bytes, which is the whole
// point of the field: the string half cannot represent a partial character.
TEST(ResponseJsonTest, ChatLogprobBytesAreUtf8) {
  ChoiceLogprobs logprobs;
  logprobs.tokens.push_back({"\xE2\x86\x92", -1.0f, {}});  // U+2192, arrow
  std::vector<ChatChoice> choices(1);
  choices[0].content = "\xE2\x86\x92";
  choices[0].logprobs = &logprobs;
  EXPECT_THAT(ChatCompletionJson("req_1", "m", choices, {1, 1}, 7),
              HasSubstr("\"bytes\":[226,134,146]"));
}

TEST(ResponseJsonTest, CompletionLogprobsAreParallelArrays) {
  ChoiceLogprobs logprobs;
  logprobs.tokens.push_back({"\xC3\xA9", -0.25f, {{"\xC3\xA9", -0.25f}}});
  logprobs.tokens.push_back({"ab", -1.0f, {}});
  std::vector<CompletionChoice> choices(1);
  choices[0].text = "hello\xC3\xA9"
                    "ab";
  choices[0].logprobs = &logprobs;
  // Echoed prompt of 5 bytes shifts every offset; the two-byte first token
  // places the second at 7.
  choices[0].text_offset = 5;
  EXPECT_THAT(
      CompletionJson("req_1", "m", choices, {2, 2}, 7),
      HasSubstr("\"logprobs\":{\"tokens\":[\"\xC3\xA9\",\"ab\"],"
                "\"token_logprobs\":[-0.250000,-1.000000],\"top_logprobs\":"
                "[{\"\xC3\xA9\":-0.250000},{}],\"text_offset\":[5,7]}"));
}

TEST(ResponseJsonTest, MultipleChoicesAreIndexedInOrder) {
  std::vector<ChatChoice> chat(2);
  chat[0].content = "first";
  chat[1].content = "second";
  chat[1].finish_reason = FinishReason::kLength;
  const std::string chat_json = ChatCompletionJson("req_1", "m", chat, {3, 4}, 7);
  EXPECT_THAT(chat_json,
              HasSubstr("{\"index\":0,\"message\":{\"role\":\"assistant\","
                        "\"content\":\"first\"},\"logprobs\":null,"
                        "\"finish_reason\":\"stop\"}"));
  EXPECT_THAT(chat_json,
              HasSubstr("{\"index\":1,\"message\":{\"role\":\"assistant\","
                        "\"content\":\"second\"},\"logprobs\":null,"
                        "\"finish_reason\":\"length\"}"));
  // Usage covers the whole response: prompt once, completions summed.
  EXPECT_THAT(chat_json, HasSubstr("\"usage\":{\"prompt_tokens\":3,"
                                   "\"completion_tokens\":4,"
                                   "\"total_tokens\":7}"));

  std::vector<CompletionChoice> text(2);
  text[0].text = "first";
  text[1].text = "second";
  const std::string text_json = CompletionJson("req_1", "m", text, {3, 4}, 7);
  EXPECT_THAT(text_json, HasSubstr("{\"index\":0,\"text\":\"first\""));
  EXPECT_THAT(text_json, HasSubstr("{\"index\":1,\"text\":\"second\""));
}

TEST(ResponseJsonTest, ChunksCarryLogprobsForTheirTokens) {
  ChoiceLogprobs logprobs;
  logprobs.tokens.push_back({"hi", -0.5f, {}});

  // A token that completed no character still owes the client its logprob
  // entry, so an empty-content chunk with a report is legal.
  const std::string chat = ChatCompletionChunkJson(
      "req_1", "m", {}, "", nullptr, 7, false, &logprobs);
  EXPECT_THAT(chat, HasSubstr("\"delta\":{\"content\":\"\"}"));
  EXPECT_THAT(chat,
              HasSubstr("\"logprobs\":{\"content\":[{\"token\":\"hi\","
                        "\"logprob\":-0.500000,\"bytes\":[104,105],"
                        "\"top_logprobs\":[]}]}"));

  const std::string text = CompletionChunkJson("req_1", "m", "hi", nullptr, 7,
                                               false, &logprobs, 12);
  EXPECT_THAT(text, HasSubstr("\"logprobs\":{\"tokens\":[\"hi\"],"
                              "\"token_logprobs\":[-0.500000],"
                              "\"top_logprobs\":[{}],\"text_offset\":[12]}"));

  // Without a report the chunk keeps its pre-logprobs bytes.
  EXPECT_THAT(CompletionChunkJson("req_1", "m", "hi", nullptr, 7),
              HasSubstr("\"logprobs\":null"));
}

}  // namespace
}  // namespace inferx::api
