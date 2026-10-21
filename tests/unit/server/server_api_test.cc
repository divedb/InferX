// Unit tests for the OpenAI-compatible API layer (src/server/api.cc):
// request parsing/validation, error objects, and SSE framing byte shapes.
// No network, no GPU.
#include "inferx/server/api.h"
#include "inferx/server/utf8.h"

#include <string>

#include "gtest/gtest.h"

namespace inferx {
namespace {

sampling::SamplingParams Defaults() {
  sampling::SamplingParams p;
  p.temperature = 1.0f;
  p.max_tokens = 16;
  return p;
}

TEST(CompletionsParseTest, AcceptsBenchShapedRequest) {
  const std::string body = R"({
    "model": "m0", "prompt": "hello",
    "repetition_penalty": 1.0, "max_tokens": 256, "logprobs": null,
    "stream": true, "stream_options": {"include_usage": true},
    "ignore_eos": true, "temperature": 0
  })";
  const ParseResult parsed = ParseCompletionsRequest(body, Defaults(), "m0");
  ASSERT_TRUE(parsed.ok) << parsed.message;
  EXPECT_EQ(parsed.request.prompt, "hello");
  EXPECT_TRUE(parsed.request.stream);
  EXPECT_EQ(parsed.request.params.max_tokens, 256u);
  EXPECT_EQ(parsed.request.params.temperature, 0.0f);
  EXPECT_TRUE(parsed.request.params.ignore_eos);
  EXPECT_EQ(parsed.request.params.repetition_penalty, 1.0f);
}

TEST(CompletionsParseTest, AppliesServerDefaultsAndPassesSamplingThrough) {
  const ParseResult parsed =
      ParseCompletionsRequest(R"({"model":"m0","prompt":"x"})", Defaults(), "m0");
  ASSERT_TRUE(parsed.ok);
  EXPECT_FALSE(parsed.request.stream);
  EXPECT_EQ(parsed.request.params.max_tokens, 16u);   // From defaults.
  EXPECT_EQ(parsed.request.params.temperature, 1.0f);
  const ParseResult seeded = ParseCompletionsRequest(
      R"({"model":"m0","prompt":"x","top_k":40,"seed":7})", Defaults(), "m0");
  ASSERT_TRUE(seeded.ok);
  EXPECT_EQ(seeded.request.params.top_k, 40u);
  ASSERT_TRUE(seeded.request.params.seed.has_value());
  EXPECT_EQ(*seeded.request.params.seed, 7u);
}

TEST(CompletionsParseTest, RejectsInvalidBodies) {
  const struct {
    std::string body;
    int status;
    std::string param;
  } bad[] = {
      {"{bad", 400, ""},
      {"[1,2]", 400, ""},
      {R"({"model":"m0"})", 400, "prompt"},
      {R"({"model":"m0","prompt":"x","max_tokens":0})", 400, "max_tokens"},
      {R"({"model":"m0","prompt":"x","temperature":"hot"})", 400, "temperature"},
      {R"({"model":"m0","prompt":"x","temperature":-1})", 400, "sampling"},
      {R"({"model":"m0","prompt":"x","top_p":0})", 400, "sampling"},
      {R"({"model":"m0","prompt":"x","logprobs":5})", 400, "logprobs"},
      {R"({"model":"m0","prompt":"x","n":2})", 400, "n"},
  };
  for (const auto& case_ : bad) {
    const ParseResult parsed = ParseCompletionsRequest(case_.body, Defaults(), "m0");
    EXPECT_FALSE(parsed.ok) << case_.body;
    EXPECT_EQ(parsed.http_status, case_.status) << case_.body;
    EXPECT_EQ(parsed.param, case_.param) << case_.body;
  }
}

TEST(CompletionsParseTest, WrongModelIsNotFound) {
  const ParseResult parsed =
      ParseCompletionsRequest(R"({"model":"gpt-4","prompt":"x"})", Defaults(), "m0");
  ASSERT_FALSE(parsed.ok);
  EXPECT_EQ(parsed.http_status, 404);
  EXPECT_EQ(parsed.code, "model_not_found");
}

TEST(ChatParseTest, ValidatesMessagesOptionsAndSampling) {
  auto parsed = ParseChatRequest(R"({"messages":[{"role":"user","content":"hi"}],
    "max_completion_tokens":8,"stop":"END","logprobs":false,
    "chat_template_kwargs":{"enable_thinking":false}})", Defaults(), "m0");
  ASSERT_TRUE(parsed.ok) << parsed.message;
  ASSERT_TRUE(parsed.request.chat.has_value());
  EXPECT_FALSE(parsed.request.chat->options.enable_thinking);
  EXPECT_EQ(parsed.request.params.max_tokens, 8);
  EXPECT_EQ(parsed.request.params.stop_strings, std::vector<std::string>{"END"});
  for (const auto* body : {
      R"({"messages":[]})", R"({"messages":[null]})",
      R"({"messages":[{"role":"user","content":[]}]})",
      R"({"messages":[{"role":"tool","content":"x"}]})",
      R"({"messages":[{"role":"assistant","content":"x","tool_calls":[]}]})",
      R"({"messages":[{"role":"user","content":"x"}],"tools":[]})",
      R"({"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"enable_thinking":1}})",
      R"({"messages":[{"role":"user","content":"x"}],"chat_template_kwargs":{"other":true}})",
      R"({"messages":[{"role":"user","content":"x"}],"max_tokens":1,"max_completion_tokens":2})"}) {
    auto invalid = ParseChatRequest(body, Defaults(), "m0");
    EXPECT_FALSE(invalid.ok) << body;
    EXPECT_EQ(invalid.http_status, 400) << body;
  }
}

TEST(CompletionsParseTest, StopControlsAreParsedAndValidated) {
  auto parsed = ParseCompletionsRequest(
      R"({"prompt":"x","stop":["END","done"],"stop_token_ids":[0,42],"ignore_eos":true})",
      Defaults(), "m0");
  ASSERT_TRUE(parsed.ok) << parsed.message;
  EXPECT_EQ(parsed.request.params.stop_strings, (std::vector<std::string>{"END", "done"}));
  EXPECT_EQ(parsed.request.params.stop_token_ids, (std::vector<std::int32_t>{0, 42}));
  EXPECT_TRUE(parsed.request.params.ignore_eos);
  for (const auto* body : {
           R"({"prompt":"x","stop":"END"})", R"({"prompt":"x","stop":null})",
           R"({"prompt":"x","stop":[],"stop_token_ids":[]})",
           R"({"prompt":"x","stop_token_ids":null})"}) {
    EXPECT_TRUE(ParseCompletionsRequest(body, Defaults(), "m0").ok) << body;
  }
  for (const auto* body : {
           R"({"prompt":"x","stop":""})", R"({"prompt":"x","stop":[""]})",
           R"({"prompt":"x","stop":3})", R"({"prompt":"x","stop":[3]})",
           R"({"prompt":"x","stop":["a","b","c","d","e"]})",
           R"({"prompt":"x","stop_token_ids":[-1]})",
           R"({"prompt":"x","stop_token_ids":[1.5]})",
           R"({"prompt":"x","stop_token_ids":[2147483648]})",
           R"({"prompt":"x","stop_token_ids":[18446744073709551615]})",
           R"({"prompt":"x","stop_token_ids":42})"}) {
    const auto invalid = ParseCompletionsRequest(body, Defaults(), "m0");
    EXPECT_FALSE(invalid.ok) << body;
    EXPECT_EQ(invalid.http_status, 400) << body;
  }
}

TEST(CompletionsParseTest, UnknownFieldsAreIgnored) {
  const ParseResult parsed = ParseCompletionsRequest(
      R"({"model":"m0","prompt":"x","future_thing":[1,2]})", Defaults(), "m0");
  EXPECT_TRUE(parsed.ok);
}

TEST(CompletionsParseTest, RejectsUnsupportedGenerationControls) {
  const Json fields = {
      {"best_of", 3},
      {"include_stop_str_in_output", true}, {"min_tokens", 4},
      {"logit_bias", {{"42", 10}}}, {"allowed_token_ids", {42}},
      {"prompt_logprobs", 2}, {"top_logprobs", 2},
      {"response_format", {{"type", "json_object"}}},
      {"structured_outputs", {{"choice", {"yes", "no"}}}},
      {"guided_json", Json::object()}, {"guided_regex", "[a-z]+"},
      {"guided_choice", {"yes", "no"}}, {"guided_grammar", "root ::= 'yes'"},
      {"guided_whitespace_pattern", " *"}, {"guided_decoding_backend", "xgrammar"},
      {"tools", Json::array()}, {"tool_choice", "auto"},
      {"parallel_tool_calls", false}, {"max_completion_tokens", 32},
      {"truncate_prompt_tokens", 8}, {"return_token_ids", true},
      {"return_tokens_as_token_ids", true}, {"use_beam_search", true},
      {"length_penalty", 1.0}, {"early_stopping", true},
      {"echo", false}, {"suffix", ""}};
  for (auto it = fields.begin(); it != fields.end(); ++it) {
    for (const Json& value : {it.value(), Json(nullptr)}) {
      Json request = {{"prompt", "x"}, {it.key(), value}};
      SCOPED_TRACE(request.dump());
      const auto parsed = ParseCompletionsRequest(request.dump(), Defaults(), "m0");
      ASSERT_FALSE(parsed.ok);
      EXPECT_EQ(parsed.http_status, 400);
      EXPECT_EQ(parsed.param, it.key());
      EXPECT_EQ(parsed.code, "unsupported_parameter");
    }
  }
}

TEST(CompletionsParseTest, EchoIsExplicitlyUnsupported) {
  // Known OpenAI fields the engine does not implement must surface as a
  // 400 unsupported_parameter, not be silently dropped
  // (docs/inference_server_design.md §2.1/§8.2).
  const ParseResult parsed =
      ParseCompletionsRequest(R"({"model":"m0","prompt":"x","echo":true})", Defaults(),
                              "m0");
  ASSERT_FALSE(parsed.ok);
  EXPECT_EQ(parsed.http_status, 400);
  EXPECT_EQ(parsed.param, "echo");
  EXPECT_EQ(parsed.code, "unsupported_parameter");
}

TEST(SseFramingTest, ChunkBytesAreExact) {
  const std::string delta_frame =
      MakeSseChunk("42", "m0", 7, "hi\n\"there\"", std::nullopt);
  EXPECT_EQ(delta_frame,
            "data: {\"id\":\"cmpl-42\",\"object\":\"text_completion_chunk\","
            "\"created\":7,\"model\":\"m0\",\"choices\":[{\"index\":0,"
            "\"text\":\"hi\\n\\\"there\\\"\",\"finish_reason\":null}]}\n\n");

  const std::string finish_frame =
      MakeSseChunk("42", "m0", 7, "", std::string_view("length"));
  EXPECT_NE(finish_frame.find("\"finish_reason\":\"length\""), std::string::npos);

  const std::string usage = MakeSseUsageChunk("42", "m0", 7, 5, 8);
  EXPECT_EQ(usage,
            "data: {\"id\":\"cmpl-42\",\"object\":\"text_completion_chunk\","
            "\"created\":7,\"model\":\"m0\",\"choices\":[],"
            "\"usage\":{\"prompt_tokens\":5,\"completion_tokens\":8,"
            "\"total_tokens\":13}}\n\n");
  EXPECT_EQ(kSseDone, "data: [DONE]\n\n");
}

TEST(Utf8StreamingTest, HoldsEveryPartialMultibyteCharacter) {
  EXPECT_EQ(detail::Utf8Boundary(""), 0u);
  EXPECT_EQ(detail::Utf8Boundary("ascii"), 5u);
  for (const std::string& character : {std::string("\xC2\xA2"),
                                      std::string("\xE2\x82\xAC"),
                                      std::string("\xF0\x9F\x98\x80")}) {
    for (std::size_t cut = 1; cut < character.size(); ++cut) {
      EXPECT_EQ(detail::Utf8Boundary(character.substr(0, cut)), 0u);
      EXPECT_EQ(detail::Utf8Boundary("prefix" + character.substr(0, cut)), 6u);
    }
    EXPECT_EQ(detail::Utf8Boundary(character), character.size());
    EXPECT_EQ(detail::Utf8Boundary("prefix" + character), 6u + character.size());
  }
}

TEST(Utf8StreamingTest, CarriesPartialBytesIntoTheNextDeltaWithoutReplacement) {
  const std::string text = "a\xC2\xA2\xE2\x82\xAC\xF0\x9F\x98\x80\nz";
  // Exercise every possible partition into deltas, including one byte per
  // delta and adjacent multibyte characters. Parse the actual SSE framing.
  for (unsigned cuts = 0; cuts < (1u << (text.size() - 1)); ++cuts) {
    std::string pending, received;
    for (std::size_t i = 0; i < text.size(); ++i) {
      pending += text[i];
      if (i + 1 < text.size() && !(cuts & (1u << i))) continue;
      const auto boundary = detail::Utf8Boundary(pending);
      const std::string delta = pending.substr(0, boundary);
      EXPECT_NO_THROW(EXPECT_FALSE(Json(delta).dump().empty()));
      const auto frame = MakeSseChunk("42", "m0", 7, delta, std::nullopt);
      const auto body = Json::parse(frame.substr(6));
      received += body["choices"][0]["text"].get<std::string>();
      pending.erase(0, boundary);
    }
    EXPECT_TRUE(pending.empty());
    EXPECT_EQ(received, text);
  }
}

TEST(SseFramingTest, ReplacesMalformedUtf8InsteadOfThrowingDuringStreaming) {
  const std::string incomplete = "\xE2\x82";
  EXPECT_THROW(EXPECT_FALSE(Json(incomplete).dump().empty()), Json::type_error);
  for (const std::string& malformed : {incomplete, std::string("\xFF"),
                                      std::string("\x80"), std::string("\xC0\xAF"),
                                      std::string("\xED\xA0\x80"),
                                      std::string("\xF4\x90\x80\x80")}) {
    std::string frame;
    ASSERT_NO_THROW(frame = MakeSseChunk("42", "m0", 7, malformed, "length"));
    const auto body = Json::parse(frame.substr(6));
    const auto text = body["choices"][0]["text"].get<std::string>();
    EXPECT_NE(text.find("\xEF\xBF\xBD"), std::string::npos);
    EXPECT_EQ(body["choices"][0]["finish_reason"], "length");
  }
}

TEST(ResponseBodyTest, ErrorAndModelShapes) {
  const Json error = MakeError(404, "no model", "model", "model_not_found");
  EXPECT_EQ(error["error"]["type"], "invalid_request_error");
  EXPECT_EQ(error["error"]["param"], "model");
  EXPECT_EQ(error["error"]["code"], "model_not_found");

  const Json models = MakeModelList("m0", 7);
  EXPECT_EQ(models["object"], "list");
  EXPECT_EQ(models["data"][0]["id"], "m0");
  EXPECT_EQ(models["data"].size(), 1u);

  const Json completion =
      MakeCompletionResponse("42", "m0", 7, "hello", "stop", 2, 5);
  EXPECT_EQ(completion["object"], "text_completion");
  EXPECT_EQ(completion["choices"][0]["text"], "hello");
  EXPECT_EQ(completion["choices"][0]["finish_reason"], "stop");
  EXPECT_EQ(completion["usage"]["total_tokens"], 7);
}

TEST(FinishReasonTest, MapsToApiSpellings) {
  EXPECT_EQ(FinishReasonName(FinishReason::kStopped), "stop");
  EXPECT_EQ(FinishReasonName(FinishReason::kLengthCapped), "length");
}

}  // namespace
}  // namespace inferx
