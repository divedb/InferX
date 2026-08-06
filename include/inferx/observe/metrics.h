#pragma once

#include <atomic>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace inferx::observe {

using Labels = std::vector<std::pair<std::string, std::string>>;

class Metric {
 public:
  enum class Type { kCounter, kGauge, kHistogram };

  virtual ~Metric() = default;
  virtual Type type() const = 0;
  virtual std::string_view name() const = 0;
  virtual std::string_view help() const = 0;
  virtual const Labels& labels() const = 0;
  virtual void AppendSamples(std::string* output) const = 0;
};

class Counter final : public Metric {
 public:
  Counter(std::string name, std::string help, Labels labels = {});

  void Increment(uint64_t amount = 1) noexcept;
  uint64_t value() const noexcept;

  Type type() const override;
  std::string_view name() const override;
  std::string_view help() const override;
  const Labels& labels() const override;
  void AppendSamples(std::string* output) const override;

 private:
  std::string name_;
  std::string help_;
  Labels labels_;
  std::atomic<uint64_t> value_{0};
};

class Gauge final : public Metric {
 public:
  Gauge(std::string name, std::string help, Labels labels = {});

  void Set(double value) noexcept;
  void Add(double amount) noexcept;
  double value() const noexcept;

  Type type() const override;
  std::string_view name() const override;
  std::string_view help() const override;
  const Labels& labels() const override;
  void AppendSamples(std::string* output) const override;

 private:
  std::string name_;
  std::string help_;
  Labels labels_;
  std::atomic<double> value_{0.0};
};

class Histogram final : public Metric {
 public:
  Histogram(std::string name, std::string help, std::vector<double> buckets,
            Labels labels = {});

  void Observe(double value) noexcept;
  /// Replaces values from an external non-cumulative histogram snapshot.
  /// Intended for scrape-time adapters, not concurrent live instruments.
  void SetSnapshot(const std::vector<uint64_t>& bucket_counts, uint64_t count,
                   double sum);
  uint64_t count() const noexcept;

  Type type() const override;
  std::string_view name() const override;
  std::string_view help() const override;
  const Labels& labels() const override;
  void AppendSamples(std::string* output) const override;

 private:
  std::string name_;
  std::string help_;
  Labels labels_;
  std::vector<double> buckets_;
  std::unique_ptr<std::atomic<uint64_t>[]> bucket_counts_;
  std::atomic<uint64_t> count_{0};
  std::atomic<double> sum_{0.0};
};

class Registry {
 public:
  std::shared_ptr<Counter> AddCounter(std::string name, std::string help,
                                      Labels labels = {});
  std::shared_ptr<Gauge> AddGauge(std::string name, std::string help,
                                  Labels labels = {});
  std::shared_ptr<Histogram> AddHistogram(std::string name, std::string help,
                                          std::vector<double> buckets,
                                          Labels labels = {});

  // Produces Prometheus text exposition format 0.0.4. Metric updates can
  // continue concurrently; each atomic sample is read once per scrape.
  std::string Render() const;

 private:
  void AddMetric(std::shared_ptr<Metric> metric);

  mutable std::mutex mutex_;
  std::vector<std::shared_ptr<Metric>> metrics_;
};

}  // namespace inferx::observe
