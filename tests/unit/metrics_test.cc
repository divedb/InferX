#include "inferx/observe/metrics.h"

#include <gtest/gtest.h>

#include <atomic>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace inferx::observe {
namespace {

TEST(MetricsTest, RendersCounterGaugeAndEscapedLabels) {
  Registry registry;
  auto counter = registry.AddCounter("inferx_requests_total",
                                     "Completed\\requests\nby outcome.",
                                     {{"outcome", "ok\"\\\n"}});
  auto gauge = registry.AddGauge("inferx_requests_running",
                                 "Requests currently executing.");
  counter->Increment(3);
  gauge->Set(2.5);

  EXPECT_EQ(registry.Render(),
            "# HELP inferx_requests_running Requests currently executing.\n"
            "# TYPE inferx_requests_running gauge\n"
            "inferx_requests_running 2.5\n"
            "# HELP inferx_requests_total Completed\\\\requests\\nby outcome.\n"
            "# TYPE inferx_requests_total counter\n"
            "inferx_requests_total{outcome=\"ok\\\"\\\\\\n\"} 3\n");
}

TEST(MetricsTest, HistogramBucketsAreCumulative) {
  Registry registry;
  auto histogram = registry.AddHistogram("inferx_step_seconds", "Step time.",
                                         {0.001, 0.01}, {{"rank", "0"}});
  histogram->Observe(0.0005);
  histogram->Observe(0.005);
  histogram->Observe(0.1);

  EXPECT_EQ(histogram->count(), 3);
  EXPECT_EQ(registry.Render(),
            "# HELP inferx_step_seconds Step time.\n"
            "# TYPE inferx_step_seconds histogram\n"
            "inferx_step_seconds_bucket{rank=\"0\",le=\"0.001\"} 1\n"
            "inferx_step_seconds_bucket{rank=\"0\",le=\"0.01\"} 2\n"
            "inferx_step_seconds_bucket{rank=\"0\",le=\"+Inf\"} 3\n"
            "inferx_step_seconds_sum{rank=\"0\"} 0.10550000000000001\n"
            "inferx_step_seconds_count{rank=\"0\"} 3\n");
}

TEST(MetricsTest, RejectsInvalidAndConflictingDescriptors) {
  Registry registry;
  EXPECT_THROW(registry.AddCounter("9bad", "Bad."), std::invalid_argument);
  registry.AddCounter("inferx_work_total", "Work.", {{"rank", "0"}});
  EXPECT_THROW(
      registry.AddCounter("inferx_work_total", "Work.", {{"rank", "0"}}),
      std::invalid_argument);
  EXPECT_THROW(registry.AddGauge("inferx_work_total", "Work.", {{"rank", "1"}}),
               std::invalid_argument);
  EXPECT_THROW(
      registry.AddHistogram("inferx_latency_seconds", "Latency.", {1.0, 0.5}),
      std::invalid_argument);
}

TEST(MetricsTest, UpdatesAndScrapesConcurrently) {
  Registry registry;
  auto counter = registry.AddCounter("inferx_events_total", "Events.");
  auto histogram = registry.AddHistogram("inferx_event_seconds", "Latency.",
                                         {0.001, 0.01, 0.1});
  constexpr int kThreads = 8;
  constexpr int kUpdates = 10000;
  std::atomic<bool> done{false};
  std::thread scraper([&] {
    while (!done.load(std::memory_order_relaxed)) {
      EXPECT_FALSE(registry.Render().empty());
    }
  });
  std::vector<std::thread> writers;
  for (int i = 0; i < kThreads; ++i) {
    writers.emplace_back([&] {
      for (int j = 0; j < kUpdates; ++j) {
        counter->Increment();
        histogram->Observe(0.005);
      }
    });
  }
  for (auto& writer : writers) writer.join();
  done.store(true, std::memory_order_relaxed);
  scraper.join();

  EXPECT_EQ(counter->value(), kThreads * kUpdates);
  EXPECT_EQ(histogram->count(), kThreads * kUpdates);
}

}  // namespace
}  // namespace inferx::observe
