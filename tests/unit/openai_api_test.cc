/// The OpenAI-compatible wire format.
///
/// This is where an HTTP API's bugs actually live -- malformed bodies, hostile
/// field types, escaping in generated text, SSE framing, stop-sequence
/// boundaries -- and none of it needs a GPU, which is why the layer is separate
/// from the engine and why this suite runs everywhere.

#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/api/openai.h"
#include "inferx/common/json.h"

namespace inferx::api {
namespace {

// Parses a built response, so assertions are about structure rather than about
// exact byte layout -- the tests should not break when a field moves.
JsonValue Parse(const std::string& text) {
  StatusOr<JsonValue> parsed = ParseJson(text);
  EXPECT_TRUE(parsed.ok()) << parsed.status().ToString() << "\nbody: " << text;

  return parsed.ok() ? *parsed : JsonValue();
}

std::string_view StringAt(const JsonValue& root,
                          std::initializer_list<std::string_view> path) {
  const JsonValue* node = &root;

  for (const std::string_view key : path) {
    if (node == nullptr) return {};

    // A numeric component indexes an array.
    if (!key.empty() && key[0] >= '0' && key[0] <= '9') {
      const StatusOr<const std::vector<JsonValue>*> array = node->AsArray();
      if (!array.ok()) return {};

      const size_t index = static_cast<size_t>(key[0] - '0');
      if (index >= (*array)->size()) return {};

      node = &(**array)[index];
      continue;
    }

    node = node->Find(key);
  }

  if (node == nullptr) return {};

  const StatusOr<std::string_view> value = node->AsString();
  return value.ok() ? *value : std::string_view{};
}

// --- Request parsing ---------------------------------------------------------

TEST(ParseChatCompletion, ReadsAWellFormedRequest) {
  const auto request = ParseChatCompletionRequest(R"({
      "model": "qwen",
      "messages": [{"role": "user", "content": "hi"}],
      "max_tokens": 32,
      "stream": true
  })");

  ASSERT_TRUE(request.ok()) << request.status().ToString();

  EXPECT_EQ(request->model, "qwen");
  ASSERT_EQ(request->messages.size(), 1u);
  EXPECT_EQ(request->messages[0].role, "user");
  EXPECT_EQ(request->messages[0].content, "hi");
  EXPECT_EQ(request->sampling.max_tokens, 32);
  EXPECT_TRUE(request->sampling.stream);
}

TEST(ParseChatCompletion, DefaultsStreamAndMaxTokens) {
  const auto request = ParseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":"hi"}]})");

  ASSERT_TRUE(request.ok()) << request.status().ToString();

  EXPECT_FALSE(request->sampling.stream);
  EXPECT_GT(request->sampling.max_tokens, 0);
}

TEST(ParseChatCompletion, AcceptsAndIgnoresSamplingParameters) {
  // Documented behaviour, not an oversight: only greedy decoding exists, and
  // rejecting these would fail nearly every client, since most send OpenAI's
  // default temperature whether the user asked for it or not.
  const auto request = ParseChatCompletionRequest(R"({
      "messages": [{"role": "user", "content": "hi"}],
      "temperature": 1.5,
      "top_p": 0.9
  })");

  EXPECT_TRUE(request.ok()) << request.status().ToString();
}

TEST(ParseChatCompletion, RejectsMalformedBodies) {
  struct Case {
    const char* body;
    const char* why;
  };

  const Case cases[] = {
      {"", "empty body"},
      {"not json", "not JSON at all"},
      {"[1,2,3]", "a JSON array rather than an object"},
      {R"({"model":"q"})", "no messages field"},
      {R"({"messages":[]})", "empty messages"},
      {R"({"messages":"hi"})", "messages is a string"},
      {R"({"messages":[{"role":"user"}]})", "message without content"},
      {R"({"messages":[{"content":"hi"}]})", "message without role"},
      {R"({"messages":[{"role":"user","content":null}]})", "null content"},
      {R"({"messages":[{"role":"user","content":5}]})", "numeric content"},
      {R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":0})",
       "max_tokens of zero"},
      {R"({"messages":[{"role":"user","content":"hi"}],"max_tokens":-1})",
       "negative max_tokens"},
      {R"({"messages":[{"role":"user","content":"hi"}],"stop":5})",
       "stop as a number"},
  };

  for (const Case& test : cases) {
    EXPECT_FALSE(ParseChatCompletionRequest(test.body).ok())
        << "accepted " << test.why << ": " << test.body;
  }
}

TEST(ParseChatCompletion, AcceptsStopAsAStringOrAnArray) {
  const auto single = ParseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":"hi"}],"stop":"END"})");

  ASSERT_TRUE(single.ok()) << single.status().ToString();
  EXPECT_EQ(single->sampling.stop, std::vector<std::string>{"END"});

  const auto many = ParseChatCompletionRequest(
      R"({"messages":[{"role":"user","content":"hi"}],"stop":["A","B"]})");

  ASSERT_TRUE(many.ok()) << many.status().ToString();
  EXPECT_EQ(many->sampling.stop, (std::vector<std::string>{"A", "B"}));
}

TEST(ParseCompletion, RejectsTheBatchPromptForm) {
  // Serving only the first element would return a response that looks complete
  // while silently dropping the rest of the batch.
  const auto request =
      ParseCompletionRequest(R"({"prompt":["one","two"]})");

  EXPECT_FALSE(request.ok());
  EXPECT_EQ(request.status().code(), absl::StatusCode::kUnimplemented);
}

TEST(ParseCompletion, ReadsASinglePrompt) {
  const auto request =
      ParseCompletionRequest(R"({"prompt":"once upon","max_tokens":8})");

  ASSERT_TRUE(request.ok()) << request.status().ToString();
  EXPECT_EQ(request->prompt, "once upon");
  EXPECT_EQ(request->sampling.max_tokens, 8);
}

// --- Response building -------------------------------------------------------

TEST(ChatCompletionJson, HasTheShapeClientsExpect) {
  Usage usage;
  usage.prompt_tokens = 11;
  usage.completion_tokens = 4;

  const JsonValue root = Parse(ChatCompletionJson(
      "chatcmpl-1", "qwen", "Hello.", FinishReason::kStop, usage, 1700000000));

  EXPECT_EQ(StringAt(root, {"object"}), "chat.completion");
  EXPECT_EQ(StringAt(root, {"id"}), "chatcmpl-1");
  EXPECT_EQ(StringAt(root, {"model"}), "qwen");
  EXPECT_EQ(StringAt(root, {"choices", "0", "message", "role"}), "assistant");
  EXPECT_EQ(StringAt(root, {"choices", "0", "message", "content"}), "Hello.");
  EXPECT_EQ(StringAt(root, {"choices", "0", "finish_reason"}), "stop");

  const StatusOr<int64_t> total =
      root.Find("usage")->RequiredInt("total_tokens");
  ASSERT_TRUE(total.ok());
  EXPECT_EQ(*total, 15);
}

TEST(ChatCompletionJson, AdvertisesGreedySampling) {
  // The only signal on the wire that temperature was ignored. If this ever
  // stops being true -- because real sampling landed -- it should stop being
  // true here too.
  Usage usage;

  EXPECT_EQ(StringAt(Parse(ChatCompletionJson("id", "m", "x",
                                              FinishReason::kStop, usage, 0)),
                     {"system_fingerprint"}),
            "greedy");
}

TEST(ChatCompletionJson, EscapesGeneratedText) {
  // Model output is arbitrary text. One unescaped quote turns a completion into
  // a parse error at the client, or lets generated text close the string and
  // forge the rest of the object.
  const std::string hostile =
      "say \"hi\"\n\tand a backslash \\ and \x01 control";

  Usage usage;
  const std::string body = ChatCompletionJson("id", "m", hostile,
                                              FinishReason::kStop, usage, 0);

  const JsonValue root = Parse(body);

  EXPECT_EQ(StringAt(root, {"choices", "0", "message", "content"}), hostile)
      << "body: " << body;
}

TEST(ChatCompletionJson, EscapesAHostileModelName) {
  Usage usage;

  const JsonValue root = Parse(ChatCompletionJson(
      "id", "a\"b", "x", FinishReason::kStop, usage, 0));

  EXPECT_EQ(StringAt(root, {"model"}), "a\"b");
}

TEST(ChatCompletionJson, KeepsMultiByteTextIntact) {
  Usage usage;

  const std::string text = "你好 🙂 café";
  const JsonValue root =
      Parse(ChatCompletionJson("id", "m", text, FinishReason::kStop, usage, 0));

  EXPECT_EQ(StringAt(root, {"choices", "0", "message", "content"}), text);
}

TEST(ChatCompletionChunk, FirstChunkCarriesRoleAndNoFinishReason) {
  const JsonValue root =
      Parse(ChatCompletionChunkJson("id", "m", "assistant", "", nullptr, 0));

  EXPECT_EQ(StringAt(root, {"object"}), "chat.completion.chunk");
  EXPECT_EQ(StringAt(root, {"choices", "0", "delta", "role"}), "assistant");

  const JsonValue* reason =
      (*root.Find("choices")->AsArray())->at(0).Find("finish_reason");
  ASSERT_NE(reason, nullptr);
  EXPECT_TRUE(reason->IsNull());
}

TEST(ChatCompletionChunk, FinalChunkHasAReasonAndNoContentKey) {
  const FinishReason reason = FinishReason::kLength;
  const JsonValue root =
      Parse(ChatCompletionChunkJson("id", "m", "", "ignored", &reason, 0));

  EXPECT_EQ(StringAt(root, {"choices", "0", "finish_reason"}), "length");

  const JsonValue* delta =
      (*root.Find("choices")->AsArray())->at(0).Find("delta");
  ASSERT_NE(delta, nullptr);
  EXPECT_EQ(delta->Find("content"), nullptr)
      << "the terminal chunk emitted a content key";
}

TEST(ErrorJson, IsParseableAndCarriesTheMessage) {
  const JsonValue root =
      Parse(ErrorJson("bad \"input\"", "invalid_request_error"));

  EXPECT_EQ(StringAt(root, {"error", "message"}), "bad \"input\"");
  EXPECT_EQ(StringAt(root, {"error", "type"}), "invalid_request_error");
}

// --- SSE framing -------------------------------------------------------------

TEST(SseFrame, IsADataLineAndABlankLine) {
  EXPECT_EQ(SseFrame("{\"a\":1}"), "data: {\"a\":1}\n\n");
}

TEST(SseFrame, SplitsAMultiLinePayloadIntoSeveralDataLines) {
  // The payloads here are single-line JSON, so this never fires in practice.
  // It is written to survive a payload that grew a newline, because the
  // alternative is one malformed event rather than a visible failure.
  EXPECT_EQ(SseFrame("a\nb"), "data: a\ndata: b\n\n");
}

TEST(SseFrame, TerminatorIsWellFormed) {
  EXPECT_EQ(SseFrame("[DONE]"), "data: [DONE]\n\n");
}

// --- Stop sequences ----------------------------------------------------------

TEST(FindStopSequence, ReturnsTheEarliestMatch) {
  const std::vector<std::string> stop = {"END", "STOP"};

  EXPECT_EQ(FindStopSequence("abc", stop), std::string::npos);
  EXPECT_EQ(FindStopSequence("abENDc", stop), 2u);

  // Earliest, not first-listed: "STOP" appears before "END" here.
  EXPECT_EQ(FindStopSequence("aSTOPbEND", stop), 1u);
}

TEST(StopSequenceHoldback, WithholdsAPartialMatchAtTheEnd) {
  const std::vector<std::string> stop = {"END"};

  // Nothing pending.
  EXPECT_EQ(StopSequenceHoldback("hello", stop), 0u);

  // "E" and "EN" could still become "END", so they cannot be emitted yet:
  // once bytes are on the wire the client has seen the start of a string it
  // explicitly asked to have removed.
  EXPECT_EQ(StopSequenceHoldback("helloE", stop), 1u);
  EXPECT_EQ(StopSequenceHoldback("helloEN", stop), 2u);

  // A complete match is FindStopSequence's business, not the holdback's, so
  // nothing is withheld here -- by the time text ends in the whole sequence,
  // the caller has already truncated at it and this is never consulted.
  EXPECT_EQ(StopSequenceHoldback("helloEND", stop), 0u);
}

TEST(StopSequenceHoldback, TakesTheLongestPendingCandidate) {
  const std::vector<std::string> stop = {"AB", "ABCD"};

  EXPECT_EQ(StopSequenceHoldback("xxxABC", stop), 3u);
}

TEST(StopSequenceHoldback, IsZeroWhenNoStopSequencesAreConfigured) {
  EXPECT_EQ(StopSequenceHoldback("anything", {}), 0u);
}

TEST(StopSequenceHoldback, HandlesTextShorterThanTheSequence) {
  const std::vector<std::string> stop = {"LONGSTOP"};

  EXPECT_EQ(StopSequenceHoldback("L", stop), 1u);
  EXPECT_EQ(StopSequenceHoldback("x", stop), 0u);
}

}  // namespace
}  // namespace inferx::api
