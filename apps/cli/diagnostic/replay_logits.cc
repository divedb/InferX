// `inferx diagnostic replay-logits` — teacher-forced numerical diagnostic.
// The replay implementation lives in src/diagnostic/replay_logits.cc.
#include <CLI/CLI.hpp>
#include <memory>
#include <string>

#include "cli/commands.h"
#include "cli/error.h"
#include "cli/app.h"
#include "inferx/diagnostic/replay_logits.h"

namespace inferx::cli {
namespace {

struct ReplayLogitsArgs {
  std::string model = std::string(kDefaultModelDir);
  std::string fixture_path;
  std::string output;
  std::string page_order = "reverse";
  std::string cache_dir;
  int chunk_size = 4096;

  diagnostic::ReplayLogitsParams Build() const {
    diagnostic::ReplayLogitsParams p;
    p.model_dir = model;
    p.fixture_path = fixture_path;
    p.output = output;
    p.page_order = page_order;
    p.cache_dir = cache_dir;
    p.chunk_size = chunk_size;
    return p;
  }
};

}  // namespace

void RegisterReplayLogits(CLI::App& parent) {
  auto args = std::make_shared<ReplayLogitsArgs>();
  CLI::App* sub = parent.add_subcommand(
      "replay-logits", "Replay identical prefixes and export final full-vocabulary FP32 logits");
  sub->add_option("--model", args->model, "Model directory")
      ->type_name("PATH")
      ->capture_default_str();
  sub->add_option("--fixture", args->fixture_path, "Replay fixture JSON")
      ->type_name("PATH")
      ->required();
  sub->add_option("--output", args->output, "FP32 logits dump path")
      ->type_name("PATH")
      ->required();
  sub->add_option("--chunk-size", args->chunk_size, "Prefill tokens per forward pass")
      ->type_name("N")
      ->capture_default_str()
      ->check(CLI::PositiveNumber);
  sub->add_option("--page-order", args->page_order, "Physical KV page permutation to exercise")
      ->type_name("NAME")
      ->capture_default_str()
      ->check(CLI::IsMember({"identity", "reverse", "shuffle"}));
  sub->add_option("--dump-cache", args->cache_dir, "Optional per-layer KV cache dump directory")
      ->type_name("PATH");
  sub->callback([args] { ThrowIfError(diagnostic::RunReplayLogits(args->Build())); });
}

}  // namespace inferx::cli
