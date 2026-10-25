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
#include "inferx/config/generation_config.h"
#include "inferx/ops/attention.h"
#include "inferx/config/merge_config.h"
#include "cli/args/engine_args.h"
#include "cli/args/model_config_args.h"
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
  model.config.model_dir = "models/foo";
  inferx::cli::EngineArgs args;
  args.max_num_seqs = 7;
  args.max_num_batched_tokens = 512;
  const ModelConfig built = model.Build();
  const CacheConfig cache = args.BuildCacheConfig();
  const SchedulerConfig scheduler = args.BuildSchedulerConfig();
  EXPECT_EQ(built.model_dir, "models/foo");
  EXPECT_EQ(cache.num_gpu_blocks_override, args.num_gpu_blocks_override);
  EXPECT_EQ(scheduler.max_num_seqs, args.max_num_seqs);
  EXPECT_EQ(scheduler.max_num_batched_tokens, args.max_num_batched_tokens);
}

TEST(EngineArgsTest, ParsesTensorParallelDevicesAndTopology) {
  inferx::cli::ModelConfigArgs model;
  inferx::cli::EngineArgs engine;
  CLI::App app{"test"};
  model.AddOptions(app);
  engine.AddOptions(app);
  Parse(app, {"--tensor-parallel-size", "2", "--device", "cuda",
              "--device-ids", "0,1"});
  EXPECT_EQ(engine.BuildParallelConfig().tensor_parallel_size, 2);
  EXPECT_EQ(engine.BuildParallelConfig().tensor_parallel_rank, 0);
  EXPECT_EQ(model.Build().device.device_ids, (std::vector<int>{0, 1}));
  EXPECT_FALSE(engine.BuildExecutionConfig().enable_cuda_graphs);
}

TEST(EngineArgsTest, ParsesVllmParityOptions) {
  inferx::cli::ModelConfigArgs model;
  inferx::cli::EngineArgs args;
  CLI::App app{"test"};
  model.AddOptions(app);
  args.AddOptions(app);
  Parse(app, {"--kv-cache-memory-bytes", "3758096384",
              "--cudagraph-capture-sizes", "1,2,4", "--max-cudagraph-capture-size", "8",
              "--no-enable-chunked-prefill"});
  const CacheConfig cache = args.BuildCacheConfig();
  const ExecutionConfig execution = args.BuildExecutionConfig();
  const SchedulerConfig scheduler = args.BuildSchedulerConfig();
  EXPECT_EQ(cache.kv_cache_memory_bytes, 3758096384);
  EXPECT_EQ(execution.cudagraph_capture_sizes, (std::vector<int>{1, 2, 4}));
  EXPECT_EQ(execution.max_cudagraph_capture_size, 8);
  EXPECT_FALSE(scheduler.enable_chunked_prefill);
}


TEST(EngineArgsTest, ChunkedPrefillDefaultsOn) {
  {
    inferx::cli::EngineArgs args;
    CLI::App app{"test"};
    args.AddOptions(app);
    EXPECT_NO_THROW(Parse(app, {}));
    EXPECT_TRUE(args.chunked_prefill);
    EXPECT_TRUE(args.BuildSchedulerConfig().enable_chunked_prefill);
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
  EXPECT_EQ(args.config.model_dir, "models/foo");
  EXPECT_EQ(args.config.tokenizer_dir, "tokenizers/foo");
  EXPECT_EQ(args.config.dtype, "float16");
  EXPECT_EQ(args.config.seed, 7);
  EXPECT_EQ(args.config.max_model_len, 2048);
  EXPECT_EQ(args.config.served_model_name, "foo");
  EXPECT_EQ(args.config.generation_config, "configs/foo");
  EXPECT_EQ(args.config.override_generation_config, R"({"temperature": 0.5})");

  const ModelConfig built = args.Build();
  EXPECT_EQ(built.model_dir, "models/foo");
  EXPECT_EQ(built.tokenizer_dir, "tokenizers/foo");
  ASSERT_TRUE(built.device.Resolve().ok()) << built.device.Resolve().status();
  EXPECT_EQ(built.device.Resolve()->front(), Device::Cuda(0));
  EXPECT_EQ(built.dtype, "float16");
  EXPECT_EQ(built.seed, 7);
  EXPECT_EQ(built.max_model_len, 2048);
}

TEST(ModelConfigArgsTest, DefaultsFallBackToModelDirAndName) {
  inferx::cli::ModelConfigArgs args;
  EXPECT_EQ(args.config.dtype, "auto");
  EXPECT_EQ(args.config.generation_config, "auto");
  EXPECT_EQ(args.config.ServedName(), args.config.model_dir);
  EXPECT_EQ(args.config.ResolvedTokenizerDir(), args.config.model_dir);
  const ModelConfig built = args.Build();
  EXPECT_EQ(built.model_dir, args.config.model_dir);
  // The fallback is derived at use, not materialized: empty means "unset".
  EXPECT_TRUE(built.tokenizer_dir.empty());
  EXPECT_EQ(built.ResolvedTokenizerDir(), args.config.model_dir);
  ASSERT_TRUE(built.device.Resolve().ok()) << built.device.Resolve().status();
  EXPECT_EQ(built.device.Resolve()->front(), Device::Cuda(0));
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

/// Writes `content` to a fresh generation config file; returns its path.
std::string WriteGenerationConfig(const std::string& name, const std::string& content) {
  const std::string dir = "/tmp/inferx_generation_config_test";
  std::filesystem::create_directories(dir);
  const std::string path = dir + "/" + name + ".json";
  { std::ofstream(path) << content; }
  return path;
}

TEST(LoadGenerationConfigTest, AutoWithoutFileIsDefaults) {
  ModelConfig config;
  config.model_dir = "/nonexistent-inferx-model-dir";  // generation_config stays "auto".
  auto loaded = LoadGenerationConfig(config);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_FLOAT_EQ(loaded->temperature, 1.0f);
  EXPECT_EQ(loaded->top_k, 0);
  EXPECT_EQ(loaded->max_new_tokens, 0);
}

TEST(LoadGenerationConfigTest, NonAutoValueIsAFilePathThatMustExist) {
  ModelConfig config;
  config.generation_config = "/nonexistent-inferx-model-dir/generation_config.json";
  EXPECT_FALSE(LoadGenerationConfig(config).ok());
  // "vllm" is not a mode here: any non-"auto" value is a path.
  config.generation_config = "vllm";
  EXPECT_FALSE(LoadGenerationConfig(config).ok());
}

TEST(LoadGenerationConfigTest, LoadsExplicitFile) {
  const std::string path =
      WriteGenerationConfig("explicit", R"({"temperature": 0.7, "top_p": 0.9})");
  ModelConfig config;
  config.model_dir = "models/foo";
  config.generation_config = path;
  auto loaded = LoadGenerationConfig(config);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_FLOAT_EQ(loaded->temperature, 0.7f);
  EXPECT_FLOAT_EQ(loaded->top_p, 0.9f);
  // Absent keys keep the struct defaults (mirroring SamplingParams).
  EXPECT_FLOAT_EQ(loaded->min_p, 0.0f);
}

TEST(LoadGenerationConfigTest, AutoLoadsFromModelDir) {
  // Its own directory: "auto" must read <model_dir>/generation_config.json
  // specifically, not any other JSON in a shared temp dir.
  const std::string dir = "/tmp/inferx_generation_config_model_dir";
  std::filesystem::create_directories(dir);
  { std::ofstream(dir + "/generation_config.json") << R"({"temperature": 0.6})"; }
  ModelConfig config;
  config.model_dir = dir;  // "auto" is the default mode.
  auto loaded = LoadGenerationConfig(config);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_FLOAT_EQ(loaded->temperature, 0.6f);
}

TEST(LoadGenerationConfigTest, OverrideMergesOnlyNamedKeys) {
  const std::string path =
      WriteGenerationConfig("override", R"({"temperature": 0.7, "top_p": 0.9})");
  ModelConfig config;
  config.generation_config = path;
  // The override replaces temperature and adds max_new_tokens; the file's
  // top_p must survive untouched.
  config.override_generation_config = R"({"temperature": 0.5, "max_new_tokens": 64})";
  auto loaded = LoadGenerationConfig(config);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_FLOAT_EQ(loaded->temperature, 0.5f);
  EXPECT_FLOAT_EQ(loaded->top_p, 0.9f);
  EXPECT_EQ(loaded->max_new_tokens, 64);

  // A malformed or non-object override is an error, not a silent no-op.
  config.override_generation_config = "{not json";
  EXPECT_FALSE(LoadGenerationConfig(config).ok());
  config.override_generation_config = "[1, 2]";
  EXPECT_FALSE(LoadGenerationConfig(config).ok());
}

TEST(LoadGenerationConfigTest, ReadsThroughConfigViewWithKeyedErrors) {
  const std::string path = WriteGenerationConfig("bad_temperature", R"({"temperature": "hot"})");
  ModelConfig config;
  config.generation_config = path;
  auto loaded = LoadGenerationConfig(config);
  ASSERT_FALSE(loaded.ok());
  // ConfigView reports the offending key in the error.
  EXPECT_NE(loaded.status().message().find("temperature"), std::string::npos);
}

TEST(LoadGenerationConfigTest, RejectsMalformedValues) {
  const auto expect_reject = [](const std::string& content) {
    const std::string path = WriteGenerationConfig("bad", content);
    ModelConfig config;
    config.generation_config = path;
    EXPECT_FALSE(LoadGenerationConfig(config).ok()) << content;
  };
  expect_reject("{not json");
  expect_reject("[1, 2]");
  expect_reject(R"({"max_new_tokens": -8})");
  expect_reject(R"({"top_k": -3})");   // Only -1 means "disabled".
  expect_reject(R"({"top_k": 2.5})");  // Non-integer.
}

TEST(LoadGenerationConfigTest, NormalizesHfDisabledTopK) {
  const std::string path = WriteGenerationConfig("topk_disabled", R"({"top_k": -1})");
  ModelConfig config;
  config.generation_config = path;
  auto loaded = LoadGenerationConfig(config);
  ASSERT_TRUE(loaded.ok()) << loaded.status();
  EXPECT_EQ(loaded->top_k, 0);
}

TEST(GenerationConfigTest, MergeAppliesValuesAndHonorsExplicitFields) {
  GenerationConfig gen;
  gen.temperature = 0.6f;
  gen.top_p = 0.95f;
  gen.top_k = 20;
  gen.repetition_penalty = 1.05f;
  gen.max_new_tokens = 512;

  sampling::SamplingParams merged;
  inferx::MergeGenerationConfig(gen, &merged);
  EXPECT_FLOAT_EQ(merged.temperature, 0.6f);
  EXPECT_FLOAT_EQ(merged.top_p, 0.95f);
  EXPECT_EQ(merged.top_k, 20u);
  EXPECT_FLOAT_EQ(merged.repetition_penalty, 1.05f);
  EXPECT_EQ(merged.max_tokens, 512u);

  // Explicitly requested fields survive the merge.
  sampling::SamplingParams explicit_wins;
  explicit_wins.temperature = 0.1f;
  explicit_wins.max_tokens = 8;
  inferx::MergeGenerationConfig(gen, &explicit_wins, {"temperature", "max-tokens"});
  EXPECT_FLOAT_EQ(explicit_wins.temperature, 0.1f);
  EXPECT_EQ(explicit_wins.max_tokens, 8u);
  EXPECT_FLOAT_EQ(explicit_wins.top_p, 0.95f);
}

TEST(GenerationConfigTest, DefaultsMergeAsNoOp) {
  // All-default config: every field mirrors the SamplingParams defaults,
  // so merging changes nothing (including max_tokens staying at its
  // engine default because max_new_tokens == 0).
  GenerationConfig gen;
  sampling::SamplingParams defaults;
  const sampling::SamplingParams untouched = defaults;
  inferx::MergeGenerationConfig(gen, &defaults);
  EXPECT_FLOAT_EQ(defaults.temperature, untouched.temperature);
  EXPECT_FLOAT_EQ(defaults.top_p, untouched.top_p);
  EXPECT_EQ(defaults.top_k, untouched.top_k);
  EXPECT_EQ(defaults.max_tokens, untouched.max_tokens);
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
