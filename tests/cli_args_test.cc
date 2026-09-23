// CLI surface tests that run in-process: build a CLI::App, parse an argv
// vector, assert on the args structs. No subprocess, no GPU.
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "CLI/CLI.hpp"
#include "gtest/gtest.h"

#include "cli/args/dataset_args.h"
#include "inferx/ops/attention.h"
#include "cli/args/engine_args.h"
#include "cli/args/model_config.h"
#include "cli/args/sampling_args.h"
#include "cli/commands.h"
#include "cli/error.h"

namespace {

using namespace inferx;

/// Parses `argv` given in reading order. CLI11's vector overload consumes
/// arguments back-to-front (the argc/argv and string overloads reverse
/// internally), so tests go through this helper.
void Parse(CLI::App& app, std::vector<std::string> argv) {
  std::reverse(argv.begin(), argv.end());
  app.parse(std::move(argv));
}

TEST(EngineArgsTest, ParsesOptionsIntoStruct) {
  inferx::cli::EngineArgs args;
  CLI::App app{"test"};
  args.AddOptions(app);
  Parse(app, {"--max-num-seqs", "4", "--block-size", "32", "--cuda-graphs"});
  EXPECT_EQ(args.max_num_seqs, 4);
  EXPECT_EQ(args.block_size, 32);
  EXPECT_TRUE(args.cuda_graphs);
}

TEST(EngineArgsTest, BuildsConsistentConfigs) {
  inferx::cli::ModelConfigArgs model;
  model.model = "models/foo";
  inferx::cli::EngineArgs args;
  args.max_num_seqs = 7;
  args.max_num_batched_tokens = 512;
  const ModelConfig built = model.Build();
  const CacheConfig cache = args.BuildCacheConfig();
  const SchedulerConfig scheduler = args.BuildSchedulerConfig();
  EXPECT_EQ(built.model_dir, "models/foo");
  EXPECT_EQ(cache.num_kv_blocks, args.num_kv_blocks);
  EXPECT_EQ(scheduler.max_num_seqs, args.max_num_seqs);
  EXPECT_EQ(scheduler.max_num_batched_tokens, args.max_num_batched_tokens);
}

TEST(EngineArgsTest, ParsesVllmParityOptions) {
  inferx::cli::ModelConfigArgs model;
  inferx::cli::EngineArgs args;
  CLI::App app{"test"};
  model.AddOptions(app);
  args.AddOptions(app);
  Parse(app, {"--kv-cache-memory-bytes", "3758096384", "--attention-backend", "flash",
              "--cudagraph-capture-sizes", "1,2,4", "--max-cudagraph-capture-size", "8",
              "--no-enable-chunked-prefill"});
  const CacheConfig cache = args.BuildCacheConfig();
  const ExecutionConfig execution = args.BuildExecutionConfig();
  const SchedulerConfig scheduler = args.BuildSchedulerConfig();
  EXPECT_EQ(cache.kv_cache_memory_bytes, 3758096384);
  EXPECT_EQ(execution.attention_backend, "flash");
  EXPECT_EQ(execution.cudagraph_capture_sizes, (std::vector<int>{1, 2, 4}));
  EXPECT_EQ(execution.max_cudagraph_capture_size, 8);
  EXPECT_FALSE(scheduler.chunked_prefill);
}

TEST(EngineArgsTest, FlashInferIsDefaultAndAliasesSelectIt) {
  inferx::cli::EngineArgs defaults;
  EXPECT_EQ(defaults.BuildExecutionConfig().attention_backend, "flashinfer");
  for (const char* name : {"flashinfer", "default", "flash"}) {
    inferx::cli::EngineArgs args;
    CLI::App app{"test"};
    args.AddOptions(app);
    EXPECT_NO_THROW(Parse(app, {"--attention-backend", name}));
    auto backend = ops::ParseAttentionBackend(args.BuildExecutionConfig().attention_backend);
    ASSERT_TRUE(backend.ok());
    EXPECT_EQ(*backend, ops::AttentionBackend::kFlashInfer);
  }
}

TEST(EngineArgsTest, ChunkedPrefillDefaultsOn) {
  {
    inferx::cli::EngineArgs args;
    CLI::App app{"test"};
    args.AddOptions(app);
    EXPECT_NO_THROW(Parse(app, {}));
    EXPECT_TRUE(args.chunked_prefill);
    EXPECT_TRUE(args.BuildSchedulerConfig().chunked_prefill);
  }
  {
    inferx::cli::EngineArgs args;
    CLI::App app{"test"};
    args.AddOptions(app);
    Parse(app, {"--enable-chunked-prefill"});
    EXPECT_TRUE(args.chunked_prefill);
  }
}

TEST(EngineArgsTest, ChunkedPrefillFlagsExcludeEachOther) {
  inferx::cli::EngineArgs args;
  CLI::App app{"test"};
  args.AddOptions(app);
  EXPECT_THROW(Parse(app, {"--enable-chunked-prefill", "--no-enable-chunked-prefill"}),
               CLI::ParseError);
}

TEST(EngineArgsTest, RejectsInvalidValues) {
  const std::vector<std::vector<std::string>> bad = {
      {"--block-size", "24"},
      {"--max-num-seqs", "0"},
      {"--attention-backend", "cutlass"},
      {"--cudagraph-capture-sizes", "0"},
      {"--max-cudagraph-capture-size", "-1"},
      {"--kv-cache-memory-bytes", "-1"}};
  for (std::vector<std::string> argv : bad) {
    inferx::cli::EngineArgs args;
    CLI::App app{"test"};
    args.AddOptions(app);
    EXPECT_THROW(Parse(app, argv), CLI::ParseError) << "expected rejection of " << argv[0];
  }
}

TEST(ModelConfigArgsTest, ParsesOptionsIntoStruct) {
  inferx::cli::ModelConfigArgs args;
  CLI::App app{"test"};
  args.AddOptions(app);
  Parse(app, {"--model", "models/foo", "--tokenizer", "tokenizers/foo", "--device", "cuda",
              "--dtype", "float16", "--seed", "7", "--max-model-len", "2048",
              "--served-model-name", "foo", "--generation-config", "configs/foo",
              "--override-generation-config", R"({"temperature": 0.5})"});
  EXPECT_EQ(args.model, "models/foo");
  EXPECT_EQ(args.tokenizer, "tokenizers/foo");
  EXPECT_EQ(args.dtype, "float16");
  EXPECT_EQ(args.seed, 7);
  EXPECT_EQ(args.max_model_len, 2048);
  EXPECT_EQ(args.served_model_name, "foo");
  EXPECT_EQ(args.generation_config, "configs/foo");
  EXPECT_EQ(args.override_generation_config, R"({"temperature": 0.5})");

  const ModelConfig built = args.Build();
  EXPECT_EQ(built.model_dir, "models/foo");
  EXPECT_EQ(built.tokenizer_dir, "tokenizers/foo");
  EXPECT_EQ(built.device, DeviceId::Cuda(0));
  EXPECT_EQ(built.dtype, "float16");
  EXPECT_EQ(built.seed, 7);
  EXPECT_EQ(built.max_model_len, 2048);
}

TEST(ModelConfigArgsTest, DefaultsFallBackToModelDirAndName) {
  inferx::cli::ModelConfigArgs args;
  EXPECT_EQ(args.dtype, "auto");
  EXPECT_EQ(args.generation_config, "auto");
  EXPECT_EQ(args.ServedName(), args.model);
  EXPECT_EQ(args.TokenizerDir(), args.model);
  const ModelConfig built = args.Build();
  EXPECT_EQ(built.model_dir, args.model);
  EXPECT_EQ(built.tokenizer_dir, args.model);
  EXPECT_EQ(built.device, DeviceId::Cuda(0));
}

TEST(ModelConfigArgsTest, RejectsInvalidValues) {
  const std::vector<std::vector<std::string>> bad = {
      {"--device", "tpu"}, {"--dtype", "fp8"}, {"--max-model-len", "-8"}};
  for (std::vector<std::string> argv : bad) {
    inferx::cli::ModelConfigArgs args;
    CLI::App app{"test"};
    args.AddOptions(app);
    EXPECT_THROW(Parse(app, argv), CLI::ParseError) << "expected rejection of " << argv[0];
  }
}

TEST(ModelConfigArgsTest, GenerationConfigResolution) {
  // "vllm" always keeps engine defaults.
  inferx::cli::ModelConfigArgs args;
  args.generation_config = "vllm";
  auto none = args.ResolveGenerationConfig();
  ASSERT_TRUE(none.ok());
  EXPECT_FALSE(none->has_value());

  // "auto" with no generation_config.json is normal, not an error.
  args.generation_config = "auto";
  args.model = "/nonexistent-inferx-model-dir";
  none = args.ResolveGenerationConfig();
  ASSERT_TRUE(none.ok());
  EXPECT_FALSE(none->has_value());

  // An explicit directory must exist and load.
  args.generation_config = "/nonexistent-inferx-model-dir";
  auto missing = args.ResolveGenerationConfig();
  EXPECT_FALSE(missing.ok());
}

TEST(ModelConfigArgsTest, GenerationConfigResolutionLoadsExplicitDirectory) {
  const std::string dir = "/tmp/inferx_generation_config_test";
  std::filesystem::create_directories(dir);
  { std::ofstream(dir + "/generation_config.json") << R"({"temperature": 0.7})"; }
  inferx::cli::ModelConfigArgs args;
  args.generation_config = dir;
  auto loaded = args.ResolveGenerationConfig();
  ASSERT_TRUE(loaded.ok());
  ASSERT_TRUE(loaded->has_value());
  ASSERT_TRUE((**loaded).temperature.has_value());
  EXPECT_FLOAT_EQ(*(**loaded).temperature, 0.7f);
}

TEST(GenerationConfigTest, FromJsonParsesAndMerges) {
  auto gen = GenerationConfig::FromJson(
      R"({"temperature": 0.6, "top_p": 0.95, "top_k": 20,)"
      R"( "repetition_penalty": 1.05, "max_new_tokens": 512})");
  ASSERT_TRUE(gen.ok());
  ASSERT_TRUE(gen->temperature.has_value());
  EXPECT_FLOAT_EQ(*gen->temperature, 0.6f);

  sampling::SamplingParams merged;
  inferx::cli::MergeGenerationConfig(*gen, &merged);
  EXPECT_FLOAT_EQ(merged.temperature, 0.6f);
  EXPECT_FLOAT_EQ(merged.top_p, 0.95f);
  EXPECT_EQ(merged.top_k, 20u);
  EXPECT_FLOAT_EQ(merged.repetition_penalty, 1.05f);
  EXPECT_EQ(merged.max_tokens, 512u);

  // Explicitly requested fields survive the merge.
  sampling::SamplingParams explicit_wins;
  explicit_wins.temperature = 0.1f;
  explicit_wins.max_tokens = 8;
  inferx::cli::MergeGenerationConfig(*gen, &explicit_wins, {"temperature", "max-tokens"});
  EXPECT_FLOAT_EQ(explicit_wins.temperature, 0.1f);
  EXPECT_EQ(explicit_wins.max_tokens, 8u);
  EXPECT_FLOAT_EQ(explicit_wins.top_p, 0.95f);
}

TEST(GenerationConfigTest, HfDisabledTopKMapsToZero) {
  auto gen = GenerationConfig::FromJson(R"({"top_k": -1})");
  ASSERT_TRUE(gen.ok());
  sampling::SamplingParams merged;
  inferx::cli::MergeGenerationConfig(*gen, &merged);
  EXPECT_EQ(merged.top_k, 0u);
}

TEST(GenerationConfigTest, EmptyJsonKeepsDefaults) {
  auto gen = GenerationConfig::FromJson("{}");
  ASSERT_TRUE(gen.ok());
  sampling::SamplingParams defaults;
  const sampling::SamplingParams untouched = defaults;
  inferx::cli::MergeGenerationConfig(*gen, &defaults);
  EXPECT_EQ(defaults.temperature, untouched.temperature);
  EXPECT_EQ(defaults.top_p, untouched.top_p);
}

TEST(SamplingArgsTest, TracksExplicitFields) {
  inferx::cli::SamplingArgs args;
  CLI::App app{"test"};
  args.AddOptions(app);
  EXPECT_TRUE(args.ExplicitFields().empty());
  Parse(app, {"--temperature", "0.3"});
  EXPECT_EQ(args.ExplicitFields(), (std::set<std::string>{"temperature"}));

  // CLI11 counts reflect the command line being parsed, so a fresh App
  // stands in for a second process passing both flags.
  inferx::cli::SamplingArgs both;
  CLI::App app2{"test"};
  both.AddOptions(app2);
  Parse(app2, {"--temperature", "0.3", "--top-p", "0.9"});
  EXPECT_EQ(both.ExplicitFields(), (std::set<std::string>{"temperature", "top-p"}));
}

TEST(DatasetArgsTest, BuildsTypedParams) {
  inferx::cli::DatasetArgs args;
  CLI::App app{"test"};
  args.AddOptions(app);
  Parse(app, {"--dataset-name", "random", "--input-len", "16"});
  const inferx::bench::DatasetParams params = args.Build();
  EXPECT_EQ(params.name, inferx::bench::DatasetName::kRandom);
  EXPECT_EQ(params.input_len, 16);
  EXPECT_TRUE(params.path.empty());
}

TEST(CommandTreeTest, DispatchesSelectedSubcommand) {
  CLI::App app{"test"};
  inferx::cli::RegisterBench(app);
  // The throughput benchmark is a stub: the callback must still fire (proving
  // dispatch works end to end) and surface the business layer's Status as a
  // CommandError.
  try {
    Parse(app, {"bench", "throughput", "--num-prompts", "4"});
    FAIL() << "expected CommandError";
  } catch (const inferx::cli::CommandError& e) {
    EXPECT_NE(std::string(e.what()).find("not implemented"), std::string::npos);
  }
}

TEST(CommandTreeTest, GroupRequiresExactlyOneChild) {
  CLI::App app{"test"};
  inferx::cli::RegisterBench(app);
  EXPECT_THROW(Parse(app, {"bench"}), CLI::RequiredError);
}

}  // namespace
