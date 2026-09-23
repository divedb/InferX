// Fixed-suite benchmark through the real scheduler and model runner; the
// CLI-independent body of `inferx bench workload`.
#include <chrono>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <cuda_profiler_api.h>

#include "inferx/bench/workload.h"
#include "inferx/core/status_util.h"
#include "inferx/engine/scheduler.h"
#include "inferx/models/model_runner.h"
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
  cc.num_kv_blocks = suite_json.value("kv_blocks", cc.num_kv_blocks);
  cc.block_size = suite_json.value("block_size", cc.block_size);
  const int batch = sc.max_num_seqs;
  auto runner = Take(ModelRunner::Create(params.model, cc, sc, params.execution));
  Scheduler scheduler(sc, runner->kv_pool(), 151645);
  uint64_t next_id = 1;
  for (const auto& c : suite_json.at("cases")) {
    if (c.at("batch") != batch) continue;
    const auto prompts = c.at("prompts").get<std::vector<std::vector<int>>>();
    const int count = prompts.size(), output_len = c.at("output_len");
    for (int repeat = -1; repeat < params.repeats; ++repeat) {
      // Complete full, identical warmup for every shape, excluding it from timing.
      auto flush = Take(scheduler.Schedule());
      Take(runner->Run(flush));
      const auto base = next_id;
      next_id += count;
      std::vector<std::vector<int>> outputs(count);
      std::vector<std::vector<double>> arrivals(count);
      std::vector<int> computed(count, 0);
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
        auto result = Take(runner->Run(plan));
        const auto run_end = Clock::now();
        if (repeat == 0 && step == params.profile_step) cudaProfilerStop();
        if (params.step_timings && repeat >= 0) {
          std::fprintf(stderr, "STEP,%s,%d,%d,%d,%zu,%.6f,%.6f\n",
              c.at("id").get<std::string>().c_str(), repeat, step,
              plan.total_num_scheduled_tokens, plan.scheduled.size(),
              std::chrono::duration<double, std::milli>(run_start - step_start).count(),
              std::chrono::duration<double, std::milli>(run_end - run_start).count());
        }
        ++step;
        Check(scheduler.UpdateFromOutput(plan, result));
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
        for (size_t j = 0; j < plan.scheduled.size(); ++j) {
          const auto& sr = plan.scheduled[j];
          const auto i = sr.request_id - base;
          computed[i] += sr.num_new_tokens;
          if (computed[i] < static_cast<int>(prompts[i].size())) continue;
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
