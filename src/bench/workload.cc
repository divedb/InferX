// Fixed-suite benchmark through the real scheduler and model runner; the
// CLI-independent body of `inferx bench workload`.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cuda_profiler_api.h>
#include <cstdlib>

#include "inferx/bench/workload.h"
#include "inferx/core/status_util.h"
#include "inferx/engine/scheduler.h"
#include "inferx/engine/model_runner.h"
#include "nlohmann/json.hpp"

namespace inferx::bench {
namespace {

using Json = nlohmann::json;
using Clock = std::chrono::steady_clock;

void Run(const WorkloadParams& params) {
  Json suite_json;
  std::ifstream(params.suite) >> suite_json;
  // Existing suites pin their capacity; honor them when present.
  SchedulerConfig sc = params.scheduler;
  CacheConfig cc = params.cache;
  sc.max_num_batched_tokens = suite_json.value("token_budget", sc.max_num_batched_tokens);
  cc.num_gpu_blocks_override =
      suite_json.value("kv_blocks", cc.num_gpu_blocks_override);
  cc.block_size = suite_json.value("block_size", cc.block_size);
  if (std::getenv("INFERX_NO_PREFIX")) cc.enable_prefix_caching = false;
  const int batch = sc.max_num_seqs;
  auto runner = Take(ModelRunner::Create(params.model, cc, sc, params.execution,
                                             params.parallel));
  Scheduler scheduler(sc, runner->GetModelSpec().eos_token_ids, *runner->Attention());
  uint64_t next_id = 1;
  for (const auto& c : suite_json.at("cases")) {
    if (c.at("batch") != batch) continue;
    const auto prompts = c.at("prompts").get<std::vector<std::vector<int>>>();
    const int count = prompts.size(), output_len = c.at("output_len");
    for (int repeat = -1; repeat < params.repeats; ++repeat) {
      // Complete full, identical warmup for every shape, excluding it from timing.
      auto flush = Take(scheduler.Schedule());
      Take(runner->ExecuteModel(flush));
      const auto base = next_id;
      next_id += count;
      std::vector<std::vector<int>> outputs(count);
      std::vector<std::vector<double>> arrivals(count);
      int step = 0;
      auto start = Clock::now();
      for (int i = 0; i < count; ++i) {
        sampling::SamplingParams p;
        p.temperature = 0;
        p.ignore_eos = true;
        p.max_tokens = output_len;
        Check(scheduler.AddRequest(Request(base + i, prompts[i], p)));
      }
      while (scheduler.HasRequests()) {
        const auto step_start = Clock::now();
        auto plan = Take(scheduler.Schedule());
        if (repeat == 0 && step == params.profile_step) cudaProfilerStart();
        const auto run_start = Clock::now();
        auto result = Take(runner->ExecuteModel(plan));
        const auto run_end = Clock::now();
        if (repeat == 0 && step == params.profile_step) cudaProfilerStop();
        if (params.step_timings && repeat >= 0) {
          std::fprintf(stderr, "STEP,%s,%d,%d,%d,%zu,%.6f,%.6f\n",
              c.at("id").get<std::string>().c_str(), repeat, step,
              plan.total_num_scheduled_tokens, plan.scheduled.size(),
              std::chrono::duration<double, std::milli>(run_start - step_start).count(),
              std::chrono::duration<double, std::milli>(run_end - run_start).count());
        }
        if (std::getenv("INFERX_PARITY_TRACE") && step < 3) {
          std::fprintf(stderr, "TRACE r=%d case=%s step=%d sched=", repeat, c.at("id").get<std::string>().c_str(), step);
          for (size_t j = 0; j < plan.scheduled.size(); ++j)
            std::fprintf(stderr, "(%llu,%d)", (unsigned long long)plan.scheduled[j].request_id,
                         plan.scheduled[j].num_new_tokens);
          std::fprintf(stderr, " samples=");
          for (size_t j = 0; j < result.samples.size(); ++j) {
            std::fprintf(stderr, "[%llu:", (unsigned long long)result.samples[j].request_id);
            for (int t : result.samples[j].token_ids) std::fprintf(stderr, "%d,", t);
            std::fprintf(stderr, "]");
          }
          std::fprintf(stderr, "\n");
        }
        ++step;
        Check(scheduler.UpdateFromOutput(plan, result));
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        for (size_t j = 0; j < plan.scheduled.size(); ++j) {
          const auto& sr = plan.scheduled[j];
          const auto i = sr.request_id - base;
          // The runner emits no samples for unfinished prefill chunks.
          // Prefix-cache hits may skip prompt tokens entirely.
          for (int t : result.samples[j].token_ids) {
            outputs[i].push_back(t);
            arrivals[i].push_back(ms);
          }
        }
        while (scheduler.PopFinished()) {
        }
      }
      const double elapsed =
          std::chrono::duration<double, std::milli>(Clock::now() - start).count();
      for (const auto& o : outputs)
        if (o.size() != static_cast<size_t>(output_len))
          throw std::runtime_error("wrong output length");
      if (repeat >= 0) {
        Json r = {{"case", c.at("id")}, {"repeat", repeat},        {"engine", "inferx"},
                  {"e2e_ms", elapsed},  {"arrivals_ms", arrivals}, {"outputs", outputs}};
        std::printf("%s\n", r.dump().c_str());
        std::fflush(stdout);
      }
    }
  }
}

}  // namespace

Status RunWorkload(const WorkloadParams& params) {
  return Guarded([&] { Run(params); });
}

}  // namespace inferx::bench
