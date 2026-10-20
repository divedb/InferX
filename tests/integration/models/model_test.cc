// End-to-end gate against the golden trace: the repository's Qwen3-0.6B
// checkpoint is loaded through the new model layer, driven by the real
// scheduler and runner, and its greedy generation must reproduce the
// recorded token stream. The one sanctioned deviation: when two candidates
// sit on the same (or adjacent) BF16 logits, argmax tie order is not
// numerically meaningful, so a first divergence is accepted only at such a
// tie, verified against the step's captured logits.
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device_runtime.h"
#include "inferx/engine/model_runner.h"
#include "inferx/engine/scheduler.h"

namespace inferx {
namespace {

std::vector<TokenId> ReadCsv(const std::string& path) {
  std::ifstream input(path);
  std::string line;
  std::getline(input, line);
  std::vector<TokenId> tokens;
  std::stringstream cells(line);
  std::string cell;
  while (std::getline(cells, cell, ',')) {
    if (!cell.empty()) tokens.push_back(std::stoi(cell));
  }
  return tokens;
}

struct Generation {
  std::unique_ptr<ModelRunner> runner;
  std::vector<TokenId> output;
};

StatusOr<Generation> Generate(const std::vector<TokenId>& prompt, int max_tokens,
                              bool capture_logits) {
  ModelConfig model;
  model.model_dir = "models/Qwen3-0.6B";
  model.device.device = "cuda";

  CacheConfig cache;
  cache.block_size = 16;
  if (std::getenv("INFERX_TEST_NO_PREFIX") != nullptr) cache.enable_prefix_caching = false;
  cache.num_kv_blocks = 1024;
  SchedulerConfig scheduler_config;
  ExecutionConfig execution;
  INFERX_ASSIGN_OR_RETURN(auto runner,
                          ModelRunner::Create(model, cache, scheduler_config,
                                                      execution));
  if (capture_logits) runner->EnableLogitsCapture();
  const models::ModelSpec& spec = runner->GetModelSpec();
  if (spec.eos_token_ids.empty()) return InternalError("checkpoint declares no eos");

  Scheduler scheduler(scheduler_config, spec.eos_token_ids, *runner->Attention());
  sampling::SamplingParams params;
  params.temperature = 0;  // Greedy: the trace was recorded greedily.
  params.max_tokens = max_tokens;
  INFERX_RETURN_IF_ERROR(scheduler.AddRequest(Request(1, prompt, params)));

  Generation generation{std::move(runner), {}};
  while (scheduler.HasRequests()) {
    INFERX_ASSIGN_OR_RETURN(SchedulerOutput plan, scheduler.Schedule());
    if (plan.IsEmpty()) continue;
    INFERX_ASSIGN_OR_RETURN(ModelRunnerOutput result, generation.runner->ExecuteModel(plan));
    INFERX_RETURN_IF_ERROR(scheduler.UpdateFromOutput(plan, result));
    for (const SampledTokens& sample : result.samples) {
      for (TokenId token : sample.token_ids) generation.output.push_back(token);
    }
  }
  return generation;
}

/// One vocabulary entry's BF16 logit, as raw bits.
StatusOr<uint16_t> LogitBits(const Tensor& logits, int64_t row, int64_t vocab_entry) {
  if (logits.GetDataType() != DataType::kBFloat16) {
    return InternalError("expected bfloat16 logits");
  }
  uint16_t bits = 0;
  INFERX_ASSIGN_OR_RETURN(DeviceRuntime* runtime, RuntimeFor(logits.GetDevice()));
  INFERX_RETURN_IF_ERROR(runtime->Memory().Copy(
      &bits, static_cast<const std::byte*>(logits.Data()) +
                 (row * logits.Dim(1) + vocab_entry) * 2,
      sizeof(bits), CopyKind::kDeviceToHost));
  return bits;
}

TEST(ModelTest, Qwen3GoldenTrace) {
  const std::vector<TokenId> prompt = ReadCsv("tests/testdata/qwen3_0_6b/prompt_tokens.txt");
  const std::vector<TokenId> expected =
      ReadCsv("tests/testdata/qwen3_0_6b/expected_tokens.txt");
  ASSERT_EQ(prompt.size(), 5u);
  ASSERT_EQ(expected.size(), 24u);

  auto run = Generate(prompt, static_cast<int>(expected.size()),
                      /*capture_logits=*/true);
  ASSERT_TRUE(run.ok()) << run.status();
  ASSERT_EQ(run->output.size(), expected.size());

  size_t first_diff = expected.size();
  for (size_t i = 0; i < expected.size(); ++i) {
    if (run->output[i] != expected[i]) {
      first_diff = i;
      break;
    }
  }
  if (first_diff == expected.size()) return;  // Exact reproduction.

  // A divergence is acceptable only when the two candidates were a BF16
  // tie (or one ulp apart) at the deciding step: argmax order between
  // equal logits is not numerically meaningful.
  ASSERT_LT(first_diff, run->runner->CapturedLogits().size())
      << "no captured logits for step " << first_diff;
  const Tensor& logits = run->runner->CapturedLogits()[first_diff];
  auto wanted = LogitBits(logits, 0, expected[first_diff]);
  auto chosen = LogitBits(logits, 0, run->output[first_diff]);
  ASSERT_TRUE(wanted.ok()) << wanted.status();
  ASSERT_TRUE(chosen.ok()) << chosen.status();
  const int ulps = std::abs(static_cast<int>(*wanted) - static_cast<int>(*chosen));
  RecordProperty("tie_ulps", ulps);
  std::printf("[golden trace] first diff at token %zu, %d ulps\n", first_diff, ulps);
  EXPECT_LE(ulps, 1) << "token " << first_diff << " diverged without a logit tie: wanted "
                     << expected[first_diff] << " (bits " << *wanted << "), chose "
                     << run->output[first_diff] << " (bits " << *chosen << ")";
}

}  // namespace
}  // namespace inferx
