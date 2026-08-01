/// Serving with CUDA graph capture turned on.
///
/// This is R9's regression test, and it lives in its own binary for a reason:
/// the bug only appeared when graphs were captured *at startup*, before the
/// model had ever run a realistic batch, which is exactly what a server does
/// and exactly what no test did. Every graph test we had captured after running
/// requests, and all of them passed throughout.
///
/// A separate file means a separate process and therefore a separate engine, so
/// this never holds two copies of the weights alongside `server_test`'s.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/server/engine.h"

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
  return !dir.empty() && std::ifstream(dir + "/config.json").good();
}

EngineConfig BaseConfig() {
  EngineConfig config;
  config.model_dir = CheckpointDir();
  // Four, not three, and not for tidiness: R9 reproduced with max_running 4
  // and did not with 3. More captured shapes than the batch actually uses is
  // the configuration that broke, so it is the one worth testing.
  config.scheduler.max_running = 4;
  config.scheduler.max_seq_len = 512;
  // Mirrors what inferx-serve computes, deliberately: the token budget is
  // larger than max_seq_len there, which means the activation buffers are
  // reserved far above what any warm-up actually fills. R9 only reproduced at
  // that ratio, so a test that quietly used a tidier number tested nothing.
  config.scheduler.max_batch_tokens = 2048;
  config.kv_blocks = 512;

  return config;
}

// Runs `prompts` concurrently and returns their continuations, in order.
std::vector<std::string> RunConcurrently(
    Engine* engine, const std::vector<std::string>& prompts,
    int32_t max_tokens) {
  std::vector<std::shared_ptr<Generation>> streams;

  for (const std::string& prompt : prompts) {
    StatusOr<std::shared_ptr<Generation>> generation = engine->Submit(
        engine->tokenizer().EncodeOrdinary(prompt), max_tokens, {});

    EXPECT_TRUE(generation.ok()) << generation.status().ToString();
    streams.push_back(generation.ok() ? *generation : nullptr);
  }

  std::vector<std::string> out;

  for (const std::shared_ptr<Generation>& stream : streams) {
    std::string text;

    if (stream != nullptr) {
      Generation::Event event;
      while (stream->Next(&event)) {
        if (event.done) break;
        text += event.text;
      }
    }

    out.push_back(std::move(text));
  }

  return out;
}

class ServerGraphsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device";
    if (!CheckpointPresent()) GTEST_SKIP() << "checkpoint not present";
  }

  const std::vector<std::string> prompts_ = {
      "The capital of Japan is",
      "The capital of Spain is",
      "The capital of France is",
  };
};

TEST_F(ServerGraphsTest, GraphedServingMatchesUngraphedServing) {
  // The property that matters: capturing a graph is an execution detail and
  // must not change a single token. Both engines are built and torn down in
  // turn rather than held at once, so this never needs two copies of the
  // weights resident.
  std::vector<std::string> ungraphed;
  std::vector<std::string> graphed;

  {
    EngineConfig config = BaseConfig();
    config.capture_graphs = false;

    StatusOr<std::unique_ptr<Engine>> engine = Engine::Create(config);
    ASSERT_TRUE(engine.ok()) << engine.status().ToString();

    ungraphed = RunConcurrently(engine->get(), prompts_, 10);
  }

  {
    EngineConfig config = BaseConfig();
    config.capture_graphs = true;

    StatusOr<std::unique_ptr<Engine>> engine = Engine::Create(config);
    ASSERT_TRUE(engine.ok()) << engine.status().ToString();

    graphed = RunConcurrently(engine->get(), prompts_, 10);
  }

  ASSERT_EQ(ungraphed.size(), prompts_.size());
  ASSERT_EQ(graphed.size(), prompts_.size());

  for (size_t i = 0; i < prompts_.size(); ++i) {
    EXPECT_EQ(graphed[i], ungraphed[i])
        << "\"" << prompts_[i]
        << "\" decoded differently once graphs were captured\n"
        << "  ungraphed: " << ungraphed[i] << "\n"
        << "  graphed  : " << graphed[i];
  }

  // A cheap sanity check that both paths produced real output rather than
  // agreeing on nothing.
  EXPECT_NE(ungraphed[0].find("Tokyo"), std::string::npos)
      << "output was: " << ungraphed[0];
}

TEST_F(ServerGraphsTest, GraphedServingIsDeterministic) {
  EngineConfig config = BaseConfig();
  config.capture_graphs = true;

  StatusOr<std::unique_ptr<Engine>> engine = Engine::Create(config);
  ASSERT_TRUE(engine.ok()) << engine.status().ToString();

  // R8 was found this way and R9 would have been too: run the same work twice
  // through one engine and require the same answer, so a result that depends on
  // what ran before it cannot pass.
  const std::vector<std::string> first =
      RunConcurrently(engine->get(), prompts_, 10);
  const std::vector<std::string> second =
      RunConcurrently(engine->get(), prompts_, 10);

  ASSERT_EQ(first.size(), second.size());

  for (size_t i = 0; i < first.size(); ++i) {
    EXPECT_EQ(first[i], second[i])
        << "\"" << prompts_[i] << "\" changed between two identical runs";
  }
}

}  // namespace
}  // namespace inferx::server
