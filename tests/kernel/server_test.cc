/// The server, end to end: HTTP in, tokens out.
///
/// M4's deliverable is "the engine is usable by something other than a test",
/// and this is the test that says so anyway -- a real socket, a real request
/// body, a real model, and the answer parsed back out of the response the way a
/// client would. Everything below the wire format is covered elsewhere; what is
/// only covered here is that the pieces are actually connected.

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "httplib.h"
#include "inferx/common/json.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/server/engine.h"
#include "inferx/server/http_server.h"

namespace inferx::server {
namespace {

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

bool CheckpointPresent() {
  const std::string dir = CheckpointDir();

  return !dir.empty() &&
         std::ifstream(dir + "/config.json").good();
}

TEST(ServerConfigTest, RejectsConflictingWeightFormatsBeforeLoading) {
  EngineConfig config;
  config.model_dir = "/path/that/must/not/be-read";
  config.fp8_weights = true;
  config.int4_weights = true;

  const StatusOr<std::unique_ptr<Engine>> created = Engine::Create(config);
  ASSERT_FALSE(created.ok());
  EXPECT_EQ(created.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(created.status().message().find("mutually exclusive"),
            std::string::npos);
}

// FP8 KV cache, end to end. A separate engine (small config, so it coexists
// with the suite's engine on the 16 GB card) built with fp8_kv_cache=true:
// warmup freezes the per-layer K/V scales, the decode graph captures the fp8
// path, and the run goes through AppendBf16AsFp8 + RunFp8/PrefillFp8 against the
// real model. The gate is the same robust fact the bf16 suite checks -- fp8 KV
// quantization error is small enough that "The capital of France is" still
// continues " Paris". A regression here means the fp8 path is wired wrong or the
// freeze did not take.
TEST(ServerTestFp8Kv, Fp8KvCacheServesTheExpectedContinuation) {
  if (!CudaAvailable() || !CheckpointPresent()) {
    GTEST_SKIP() << "needs a CUDA device and the test checkpoint";
  }

  EngineConfig config;
  config.model_dir = CheckpointDir();
  config.scheduler.max_running = 2;
  config.scheduler.max_seq_len = 256;
  config.scheduler.max_batch_tokens = 256;
  config.kv_blocks = 128;
  config.fp8_kv_cache = true;
  // Graphs stay on: capturing the fp8 decode path is part of what this tests.

  StatusOr<std::unique_ptr<Engine>> created = Engine::Create(config);
  ASSERT_TRUE(created.ok()) << created.status().ToString();
  std::unique_ptr<Engine> engine = std::move(*created);

  const std::vector<int32_t> prompt =
      engine->tokenizer().EncodeOrdinary("The capital of France is");

  StatusOr<std::shared_ptr<Generation>> generation =
      engine->Submit(prompt, /*max_tokens=*/4, /*stop=*/{});
  ASSERT_TRUE(generation.ok()) << generation.status().ToString();

  std::string text;
  Generation::Event event;
  while ((*generation)->Next(&event)) {
    if (event.done) break;
    text += event.text;
  }

  EXPECT_TRUE(text.rfind(" Paris", 0) == 0)
      << "expected the fp8-KV continuation to start with \" Paris\", got: "
      << text;
}

TEST(ServerTestW4A16, Int4WeightsServeTheExpectedContinuation) {
  if (!CudaAvailable() || !CheckpointPresent()) {
    GTEST_SKIP() << "needs a CUDA device and the test checkpoint";
  }

  EngineConfig config;
  config.model_dir = CheckpointDir();
  config.scheduler.max_running = 2;
  config.scheduler.max_seq_len = 256;
  config.scheduler.max_batch_tokens = 256;
  config.kv_blocks = 128;
  config.int4_weights = true;

  StatusOr<std::unique_ptr<Engine>> created = Engine::Create(config);
  ASSERT_TRUE(created.ok()) << created.status().ToString();
  std::unique_ptr<Engine> engine = std::move(*created);

  const std::vector<int32_t> prompt =
      engine->tokenizer().EncodeOrdinary("The capital of France is");
  StatusOr<std::shared_ptr<Generation>> generation =
      engine->Submit(prompt, /*max_tokens=*/4, /*stop=*/{});
  ASSERT_TRUE(generation.ok()) << generation.status().ToString();

  std::string text;
  Generation::Event event;
  while ((*generation)->Next(&event)) {
    if (event.done) break;
    text += event.text;
  }

  EXPECT_TRUE(text.rfind(" Paris", 0) == 0)
      << "expected the W4A16 continuation to start with \" Paris\", got: "
      << text;
}

// The engine is expensive to build -- weights, KV pool, warm-up -- so the whole
// suite shares one, and every test is written to leave it usable by the next.
class ServerTest : public ::testing::Test {
 protected:
  static void SetUpTestSuite() {
    if (!CudaAvailable() || !CheckpointPresent()) return;

    EngineConfig config;
    config.model_dir = CheckpointDir();
    config.scheduler.max_running = 4;
    config.scheduler.max_seq_len = 512;
    config.scheduler.max_batch_tokens = 512;
    config.kv_blocks = 512;

    // Left at the default, which is now on: R9 is fixed, so the served
    // configuration and the tested one are the same again.

    StatusOr<std::unique_ptr<Engine>> created = Engine::Create(config);
    ASSERT_TRUE(created.ok()) << created.status().ToString();

    engine_ = created->release();

    HttpServerConfig http;
    http.host = "127.0.0.1";
    http.port = 0;  // any free port, so the suite cannot collide

    StatusOr<std::unique_ptr<HttpServer>> server =
        HttpServer::Create(engine_, http);
    ASSERT_TRUE(server.ok()) << server.status().ToString();

    server_ = server->release();

    listener_ = new std::thread([] { (void)server_->Listen(); });

    ASSERT_TRUE(server_->WaitUntilReady());
    ASSERT_GT(server_->port(), 0);
  }

  static void TearDownTestSuite() {
    if (server_ != nullptr) server_->Stop();

    if (listener_ != nullptr) {
      listener_->join();
      delete listener_;
      listener_ = nullptr;
    }

    delete server_;
    server_ = nullptr;

    delete engine_;
    engine_ = nullptr;
  }

  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    if (!CheckpointPresent()) GTEST_SKIP() << "checkpoint not present";
  }

  static httplib::Client Client() {
    httplib::Client client("127.0.0.1", server_->port());
    client.set_read_timeout(120, 0);

    return client;
  }

  static JsonValue ParseBody(const std::string& body) {
    StatusOr<JsonValue> parsed = ParseJson(body);
    EXPECT_TRUE(parsed.ok()) << parsed.status().ToString() << "\n" << body;

    return parsed.ok() ? *parsed : JsonValue();
  }

  static Engine* engine_;
  static HttpServer* server_;
  static std::thread* listener_;
};

Engine* ServerTest::engine_ = nullptr;
HttpServer* ServerTest::server_ = nullptr;
std::thread* ServerTest::listener_ = nullptr;

// --- The engine, without HTTP in the way -------------------------------------

TEST_F(ServerTest, EngineGeneratesTheExpectedContinuation) {
  // Greedy decoding is deterministic, so this is an equality assertion rather
  // than a "looks plausible" one. "The capital of France is" continues
  // " Paris" for any model that has learned anything at all, and if it stops
  // doing so the engine is broken, not the model.
  const std::vector<int32_t> prompt =
      engine_->tokenizer().EncodeOrdinary("The capital of France is");

  StatusOr<std::shared_ptr<Generation>> generation =
      engine_->Submit(prompt, /*max_tokens=*/4, /*stop=*/{});
  ASSERT_TRUE(generation.ok()) << generation.status().ToString();

  std::string text;
  Generation::Event event;

  while ((*generation)->Next(&event)) {
    if (event.done) break;
    text += event.text;
  }

  EXPECT_TRUE(text.rfind(" Paris", 0) == 0)
      << "expected the continuation to start with \" Paris\", got: " << text;
}

TEST_F(ServerTest, ConcurrentRequestsDoNotContaminateEachOther) {
  // Batched decode shares a step, a KV pool and a block table, so a bug in any
  // of them shows up as one sequence's tokens appearing in another's output.
  struct Case {
    std::string prompt;
    std::string expect;
  };

  const std::vector<Case> cases = {
      {"The capital of France is", " Paris"},
      {"The capital of Japan is", " Tokyo"},
      {"The capital of Spain is", " Madrid"},
  };

  std::vector<std::shared_ptr<Generation>> streams;

  for (const Case& test : cases) {
    StatusOr<std::shared_ptr<Generation>> generation = engine_->Submit(
        engine_->tokenizer().EncodeOrdinary(test.prompt), 4, {});

    ASSERT_TRUE(generation.ok()) << generation.status().ToString();
    streams.push_back(*generation);
  }

  for (size_t i = 0; i < streams.size(); ++i) {
    std::string text;
    Generation::Event event;

    while (streams[i]->Next(&event)) {
      if (event.done) break;
      text += event.text;
    }

    EXPECT_TRUE(text.rfind(cases[i].expect, 0) == 0)
        << "request " << i << " (" << cases[i].prompt << ") expected \""
        << cases[i].expect << "\", got: " << text;
  }
}

TEST_F(ServerTest, CancellingStopsGenerationEarly) {
  StatusOr<std::shared_ptr<Generation>> generation = engine_->Submit(
      engine_->tokenizer().EncodeOrdinary("Count upwards forever:"),
      /*max_tokens=*/200, {});
  ASSERT_TRUE(generation.ok()) << generation.status().ToString();

  Generation::Event event;

  // Read a little, then walk away, which is what a disconnect looks like from
  // the engine's side.
  int chunks = 0;
  while ((*generation)->Next(&event) && !event.done) {
    if (++chunks >= 3) break;
  }

  (*generation)->Cancel();

  // The stream must terminate rather than hang: a client that disconnects must
  // not leave a sequence occupying KV blocks forever.
  const auto deadline = std::chrono::steady_clock::now() +
                        std::chrono::seconds(30);

  bool finished = false;

  while (std::chrono::steady_clock::now() < deadline) {
    if (!(*generation)->Next(&event)) {
      finished = true;
      break;
    }

    if (event.done) {
      finished = true;
      break;
    }
  }

  EXPECT_TRUE(finished) << "a cancelled generation never terminated";

  // And the blocks come back.
  const auto blocks_deadline =
      std::chrono::steady_clock::now() + std::chrono::seconds(30);

  while (std::chrono::steady_clock::now() < blocks_deadline) {
    if (engine_->stats().running == 0) break;
    std::this_thread::sleep_for(std::chrono::milliseconds(20));
  }

  EXPECT_EQ(engine_->stats().running, 0)
      << "a cancelled sequence is still running";
}

TEST_F(ServerTest, IdenticalRequestsProduceIdenticalOutput) {
  // Sampling is greedy, so repeating a prompt must repeat its tokens -- a
  // request's output must not depend on what ran before it.
  //
  // This was R8. Consecutive requests alternated between two continuations
  // because `PrepareStep` copied a sequence's block-table row before `Reserve`
  // grew it, so on the one step where a sequence first reached into a new
  // block, attention was sent to block 0 instead. Whether that mattered
  // depended on which physical block the stack free list had handed out, which
  // is why the symptom was alternation rather than a steady error.
  //
  // Kept at the server level even though the cause was in the scheduler and is
  // now covered host-side by SchedulerTest.EveryBatchSlotIsCoveredByTheBatch-
  // BlockTable: this is the property a user actually cares about, and it is
  // worth asserting where they would notice it breaking.
  //
  // **The first response is excluded, and that is a real concession.** With
  // prefix caching on (§6.3), the first request computes its prompt and every
  // request after it reads that prompt's KV instead, forwarding only the last
  // block. Fewer rows means cuBLASLt may pick a different algorithm, bf16
  // reductions do not commute, and the logits land a fraction apart -- enough
  // to flip a genuinely close call. On this prompt it does: cold answers
  // "Green", warm answers "Yellow", every time.
  //
  // So a cache hit can change a greedy answer. That is a property of prefix
  // caching rather than a defect in this one, and every engine that does it has
  // it; `enable_prefix_cache = false` buys the stronger guarantee back. What
  // must still hold, and what this asserts, is that the answer is stable for a
  // given cache state -- which is what R8 violated, since alternation between
  // consecutive warm requests would fail here exactly as it did before.
  const char* body =
      R"({"messages":[{"role":"user","content":"Name three colours."}],)"
      R"("max_tokens":32})";

  httplib::Client client = Client();

  std::vector<std::string> answers;

  for (int attempt = 0; attempt < 6; ++attempt) {
    const httplib::Result response =
        client.Post("/v1/chat/completions", body, "application/json");

    ASSERT_TRUE(response);
    ASSERT_EQ(response->status, 200) << response->body;

    const JsonValue root = ParseBody(response->body);
    const StatusOr<std::string_view> content =
        (**root.Find("choices")->AsArray())[0]
            .Find("message")
            ->RequiredString("content");
    ASSERT_TRUE(content.ok());

    answers.emplace_back(*content);
  }

  ASSERT_GE(answers.size(), 3u);

  // answers[0] warmed the cache. Everything from answers[1] on sees the same
  // cached prefix as everything else, so they must agree exactly.
  for (size_t i = 2; i < answers.size(); ++i) {
    EXPECT_EQ(answers[i], answers[1])
        << "greedy decoding returned a different answer on attempt " << i
        << " with the cache in the same state;\n  first warm: " << answers[1]
        << "\n  this      : " << answers[i];
  }
}

TEST_F(ServerTest, RejectsAPromptThatLeavesNoRoomToGenerate) {
  const std::vector<int32_t> prompt(1000, 100);

  EXPECT_FALSE(engine_->Submit(prompt, 8, {}).ok())
      << "accepted a prompt longer than max_seq_len";
}

// --- Over HTTP ---------------------------------------------------------------

TEST_F(ServerTest, HealthAndModelsRespond) {
  httplib::Client client = Client();

  const httplib::Result health = client.Get("/health");
  ASSERT_TRUE(health) << "no response";
  EXPECT_EQ(health->status, 200);

  const httplib::Result models = client.Get("/v1/models");
  ASSERT_TRUE(models);
  EXPECT_EQ(models->status, 200);

  const JsonValue root = ParseBody(models->body);
  const StatusOr<std::string_view> object = root.RequiredString("object");
  ASSERT_TRUE(object.ok());
  EXPECT_EQ(*object, "list");
}

TEST_F(ServerTest, ChatCompletionReturnsAnAnswer) {
  httplib::Client client = Client();

  const httplib::Result response = client.Post(
      "/v1/chat/completions",
      R"({"messages":[{"role":"user","content":)"
      R"("What is the capital of France? Answer in one word."}],)"
      R"("max_tokens":16})",
      "application/json");

  ASSERT_TRUE(response) << "no response";
  ASSERT_EQ(response->status, 200) << response->body;

  const JsonValue root = ParseBody(response->body);

  const JsonValue* choices = root.Find("choices");
  ASSERT_NE(choices, nullptr);

  const StatusOr<const std::vector<JsonValue>*> list = choices->AsArray();
  ASSERT_TRUE(list.ok());
  ASSERT_EQ((*list)->size(), 1u);

  const StatusOr<std::string_view> content =
      (**list)[0].Find("message")->RequiredString("content");
  ASSERT_TRUE(content.ok());

  EXPECT_NE(content->find("Paris"), std::string_view::npos)
      << "answer was: " << *content;

  // The chat template ends the turn on <|im_end|>, so a one-word answer should
  // stop rather than run to the token limit.
  const StatusOr<std::string_view> reason =
      (**list)[0].RequiredString("finish_reason");
  ASSERT_TRUE(reason.ok());
  EXPECT_EQ(*reason, "stop");
}

TEST_F(ServerTest, MalformedRequestsGet400WithAnOpenAiShapedError) {
  httplib::Client client = Client();

  for (const char* body : {"", "not json", R"({"messages":[]})",
                           R"({"messages":[{"role":"user"}]})"}) {
    const httplib::Result response =
        client.Post("/v1/chat/completions", body, "application/json");

    ASSERT_TRUE(response) << "no response for body: " << body;
    EXPECT_EQ(response->status, 400) << "body: " << body;

    const JsonValue root = ParseBody(response->body);
    EXPECT_NE(root.Find("error"), nullptr) << response->body;
  }
}

TEST_F(ServerTest, StreamingDeltasConcatenateToTheBlockingAnswer) {
  // The two endpoints must agree. If they do not, a completion read over SSE
  // differs from the same completion read in one piece, which gets reported as
  // a model bug and is not one.
  httplib::Client client = Client();

  constexpr const char* kBody =
      R"({"messages":[{"role":"user","content":"Name three colours."}],)"
      R"("max_tokens":32})";

  // Warm the prefix cache before comparing anything. Without this the blocking
  // request below computes its prompt and the streaming one reads the KV that
  // request left behind, so the two are being asked to agree across a change
  // in how the arithmetic is arranged rather than across a change of endpoint
  // -- and on a close call they do not (§6.3). Each test case runs in its own
  // process here, so the cache really does start empty every time.
  (void)client.Post("/v1/chat/completions", kBody, "application/json");

  const httplib::Result blocking =
      client.Post("/v1/chat/completions", kBody, "application/json");
  ASSERT_TRUE(blocking);
  ASSERT_EQ(blocking->status, 200) << blocking->body;

  const JsonValue root = ParseBody(blocking->body);
  const StatusOr<std::string_view> expected =
      (**root.Find("choices")->AsArray())[0]
          .Find("message")
          ->RequiredString("content");
  ASSERT_TRUE(expected.ok());

  const std::string streaming_body =
      std::string(R"({"messages":[{"role":"user","content":)"
                  R"("Name three colours."}],"max_tokens":32,"stream":true})");

  std::string raw;
  const httplib::Result streamed = client.Post(
      "/v1/chat/completions", streaming_body, "application/json");

  ASSERT_TRUE(streamed);
  ASSERT_EQ(streamed->status, 200);
  raw = streamed->body;

  // Reassemble the deltas the way a client would.
  std::string assembled;
  bool saw_done = false;
  bool saw_finish_reason = false;

  size_t at = 0;
  while ((at = raw.find("data: ", at)) != std::string::npos) {
    at += 6;

    const size_t end = raw.find('\n', at);
    const std::string payload =
        raw.substr(at, end == std::string::npos ? end : end - at);

    if (payload == "[DONE]") {
      saw_done = true;
      break;
    }

    const JsonValue chunk = ParseBody(payload);
    const JsonValue* choice = &(**chunk.Find("choices")->AsArray())[0];

    if (const JsonValue* reason = choice->Find("finish_reason");
        reason != nullptr && !reason->IsNull()) {
      saw_finish_reason = true;
    }

    if (const JsonValue* delta = choice->Find("delta"); delta != nullptr) {
      if (const JsonValue* content = delta->Find("content");
          content != nullptr) {
        if (const StatusOr<std::string_view> text = content->AsString();
            text.ok()) {
          assembled += *text;
        }
      }
    }
  }

  EXPECT_TRUE(saw_done) << "the stream never sent [DONE]";
  EXPECT_TRUE(saw_finish_reason) << "no chunk carried a finish_reason";
  EXPECT_EQ(assembled, *expected);
}

TEST_F(ServerTest, StreamingUsageChunkMatchesTheBlockingCounts) {
  // A streaming client's only other way to learn its token counts is to count
  // chunks, and that is wrong whenever a token decodes to no complete
  // character -- the server skips those chunks. So the counts reported here
  // have to be the engine's, and the blocking endpoint is what says what the
  // engine's are. M10's benchmark divides by these numbers.
  httplib::Client client = Client();

  constexpr const char* kPrompt =
      R"("prompt":"The capital of France is","max_tokens":24)";

  const httplib::Result blocking = client.Post(
      "/v1/completions", std::string("{") + kPrompt + "}", "application/json");
  ASSERT_TRUE(blocking);
  ASSERT_EQ(blocking->status, 200) << blocking->body;

  const JsonValue expected = ParseBody(blocking->body);
  const JsonValue* expected_usage = expected.Find("usage");
  ASSERT_NE(expected_usage, nullptr);

  const StatusOr<int64_t> expected_prompt =
      expected_usage->RequiredInt("prompt_tokens");
  const StatusOr<int64_t> expected_completion =
      expected_usage->RequiredInt("completion_tokens");
  ASSERT_TRUE(expected_prompt.ok());
  ASSERT_TRUE(expected_completion.ok());

  const httplib::Result streamed = client.Post(
      "/v1/completions",
      std::string("{") + kPrompt +
          R"(,"stream":true,"stream_options":{"include_usage":true}})",
      "application/json");
  ASSERT_TRUE(streamed);
  ASSERT_EQ(streamed->status, 200);

  // Walk to the last chunk before [DONE], which is where the usage chunk has
  // to be: a client that reads only up to the finish reason must still get a
  // well-formed stream, so the counts cannot come any earlier.
  const std::string& raw = streamed->body;
  int usage_chunks = 0;
  int64_t prompt_tokens = -1;
  int64_t completion_tokens = -1;
  bool usage_was_last = false;

  size_t at = 0;
  while ((at = raw.find("data: ", at)) != std::string::npos) {
    at += 6;

    const size_t end = raw.find('\n', at);
    const std::string payload =
        raw.substr(at, end == std::string::npos ? end : end - at);

    if (payload == "[DONE]") break;

    const JsonValue chunk = ParseBody(payload);
    const StatusOr<const std::vector<JsonValue>*> choices =
        chunk.Find("choices")->AsArray();
    ASSERT_TRUE(choices.ok());

    if (!(*choices)->empty()) {
      usage_was_last = false;
      continue;
    }

    ++usage_chunks;
    usage_was_last = true;

    const JsonValue* usage = chunk.Find("usage");
    ASSERT_NE(usage, nullptr) << "a choice-less chunk carried no usage";

    const StatusOr<int64_t> p = usage->RequiredInt("prompt_tokens");
    const StatusOr<int64_t> c = usage->RequiredInt("completion_tokens");
    ASSERT_TRUE(p.ok());
    ASSERT_TRUE(c.ok());
    prompt_tokens = *p;
    completion_tokens = *c;
  }

  EXPECT_EQ(usage_chunks, 1) << "expected exactly one usage chunk";
  EXPECT_TRUE(usage_was_last) << "the usage chunk was not the last before [DONE]";
  EXPECT_EQ(prompt_tokens, *expected_prompt);
  EXPECT_EQ(completion_tokens, *expected_completion);
}

TEST_F(ServerTest, StreamingOmitsUsageUnlessAsked) {
  // The extra chunk is opt-in: a client that never sent stream_options must
  // see the stream it saw before this feature existed.
  httplib::Client client = Client();

  const httplib::Result streamed = client.Post(
      "/v1/completions",
      R"({"prompt":"The capital of France is","max_tokens":8,"stream":true})",
      "application/json");
  ASSERT_TRUE(streamed);
  ASSERT_EQ(streamed->status, 200);

  EXPECT_EQ(streamed->body.find("\"usage\""), std::string::npos)
      << "an unasked-for usage chunk appeared in the stream";
}

TEST_F(ServerTest, StopStringIsRemovedFromTheOutput) {
  httplib::Client client = Client();

  const httplib::Result response = client.Post(
      "/v1/completions",
      R"({"prompt":"The capital of France is","max_tokens":32,)"
      R"("stop":["Germany"]})",
      "application/json");

  ASSERT_TRUE(response);
  ASSERT_EQ(response->status, 200) << response->body;

  const JsonValue root = ParseBody(response->body);
  const JsonValue* choice = &(**root.Find("choices")->AsArray())[0];

  const StatusOr<std::string_view> text = choice->RequiredString("text");
  ASSERT_TRUE(text.ok());

  // The stop string must not appear, not even partially at the tail -- that is
  // what the holdback in the streaming path is for.
  EXPECT_EQ(text->find("Germany"), std::string_view::npos)
      << "the stop string survived into the output: " << *text;

  const StatusOr<std::string_view> reason =
      choice->RequiredString("finish_reason");
  ASSERT_TRUE(reason.ok());
  EXPECT_EQ(*reason, "stop");
}

TEST_F(ServerTest, MetricsReportTheEnginesCounters) {
  httplib::Client client = Client();

  const httplib::Result response = client.Get("/metrics");
  ASSERT_TRUE(response);
  ASSERT_EQ(response->status, 200);
  EXPECT_NE(response->get_header_value("Content-Type").find("text/plain"),
            std::string::npos);
  for (const char* metric : {"inferx_requests_running ",
                             "inferx_requests_waiting ",
                             "inferx_kv_blocks{state=\"used\"} ",
                             "inferx_steps_total ",
                             "inferx_generation_tokens_total "}) {
    EXPECT_NE(response->body.find(metric), std::string::npos)
        << "/metrics is missing \"" << metric << "\": " << response->body;
  }
  auto sample = [](std::string_view body, std::string_view metric) {
    const size_t start = body.find(std::string(metric) + " ");
    if (start == std::string_view::npos) return int64_t{-1};
    return static_cast<int64_t>(
        std::stoll(std::string(body.substr(start + metric.size() + 1))));
  };
  const int64_t before = sample(response->body, "inferx_steps_total");
  ASSERT_GE(before, 0);

  const httplib::Result stats = client.Get("/stats");
  ASSERT_TRUE(stats);
  EXPECT_TRUE(ParseBody(stats->body).RequiredInt("steps").ok());

  // Generate something *here* rather than relying on earlier tests: ctest runs
  // each test in its own process, so a counter that looks warm when the whole
  // binary runs is zero when the case runs alone.
  const httplib::Result generated = client.Post(
      "/v1/completions", R"({"prompt":"Hello","max_tokens":4})",
      "application/json");

  ASSERT_TRUE(generated);
  ASSERT_EQ(generated->status, 200) << generated->body;

  const httplib::Result after_response = client.Get("/metrics");
  ASSERT_TRUE(after_response);

  const int64_t after = sample(after_response->body, "inferx_steps_total");
  ASSERT_GE(after, 0);
  EXPECT_GT(after, before) << "the step counter did not advance";
}

}  // namespace
}  // namespace inferx::server
