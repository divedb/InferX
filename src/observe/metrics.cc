#include "inferx/observe/metrics.h"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <mutex>
#include <stdexcept>

namespace inferx::observe {
namespace {

bool IsMetricFirst(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' ||
         c == ':';
}

bool IsMetricRest(char c) { return IsMetricFirst(c) || (c >= '0' && c <= '9'); }

bool IsLabelFirst(char c) {
  return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_';
}

void ValidateName(std::string_view value, bool label) {
  const auto first = label ? IsLabelFirst : IsMetricFirst;
  const auto rest =
      label ? [](char c) { return IsLabelFirst(c) || (c >= '0' && c <= '9'); }
            : IsMetricRest;
  if (value.empty() || !first(value.front()) ||
      !std::all_of(value.begin() + 1, value.end(), rest)) {
    throw std::invalid_argument(label ? "invalid Prometheus label name"
                                      : "invalid Prometheus metric name");
  }
}

void ValidateDescriptor(std::string_view name, const Labels& labels) {
  ValidateName(name, false);
  std::vector<std::string_view> names;
  names.reserve(labels.size());
  for (const auto& [key, value] : labels) {
    (void)value;
    ValidateName(key, true);
    if (std::find(names.begin(), names.end(), key) != names.end()) {
      throw std::invalid_argument("duplicate Prometheus label name");
    }
    names.push_back(key);
  }
}

void AppendEscaped(std::string_view value, std::string* output, bool help) {
  for (char c : value) {
    if (c == '\\' || c == '\n' || (!help && c == '"')) output->push_back('\\');
    output->push_back(c == '\n' ? 'n' : c);
  }
}

void AppendDouble(double value, std::string* output) {
  if (std::isnan(value)) {
    output->append("NaN");
    return;
  }
  if (std::isinf(value)) {
    output->append(value < 0 ? "-Inf" : "+Inf");
    return;
  }
  char buffer[64];
  auto [end, error] = std::to_chars(buffer, buffer + sizeof(buffer), value);
  if (error == std::errc()) output->append(buffer, end);
}

void AppendLabels(const Labels& labels, std::string_view extra_name,
                  std::string_view extra_value, std::string* output) {
  if (labels.empty() && extra_name.empty()) return;
  output->push_back('{');
  bool first = true;
  auto append = [&](std::string_view name, std::string_view value) {
    if (!first) output->push_back(',');
    first = false;
    output->append(name);
    output->append("=\"");
    AppendEscaped(value, output, false);
    output->push_back('"');
  };
  for (const auto& [name, value] : labels) append(name, value);
  if (!extra_name.empty()) append(extra_name, extra_value);
  output->push_back('}');
}

std::string TypeName(Metric::Type type) {
  switch (type) {
    case Metric::Type::kCounter:
      return "counter";
    case Metric::Type::kGauge:
      return "gauge";
    case Metric::Type::kHistogram:
      return "histogram";
  }
  return "untyped";
}

std::string LabelKey(const Labels& labels) {
  std::string result;
  for (const auto& [name, value] : labels) {
    result.append(name).push_back('\0');
    result.append(value).push_back('\0');
  }
  return result;
}

}  // namespace

Counter::Counter(std::string name, std::string help, Labels labels)
    : name_(std::move(name)),
      help_(std::move(help)),
      labels_(std::move(labels)) {
  std::sort(labels_.begin(), labels_.end());
  ValidateDescriptor(name_, labels_);
}

void Counter::Increment(uint64_t amount) noexcept {
  value_.fetch_add(amount, std::memory_order_relaxed);
}
uint64_t Counter::value() const noexcept {
  return value_.load(std::memory_order_relaxed);
}
Metric::Type Counter::type() const { return Type::kCounter; }
std::string_view Counter::name() const { return name_; }
std::string_view Counter::help() const { return help_; }
const Labels& Counter::labels() const { return labels_; }
void Counter::AppendSamples(std::string* output) const {
  output->append(name_);
  AppendLabels(labels_, {}, {}, output);
  output->push_back(' ');
  output->append(std::to_string(value()));
  output->push_back('\n');
}

Gauge::Gauge(std::string name, std::string help, Labels labels)
    : name_(std::move(name)),
      help_(std::move(help)),
      labels_(std::move(labels)) {
  std::sort(labels_.begin(), labels_.end());
  ValidateDescriptor(name_, labels_);
}
void Gauge::Set(double value) noexcept {
  value_.store(value, std::memory_order_relaxed);
}
void Gauge::Add(double amount) noexcept {
  value_.fetch_add(amount, std::memory_order_relaxed);
}
double Gauge::value() const noexcept {
  return value_.load(std::memory_order_relaxed);
}
Metric::Type Gauge::type() const { return Type::kGauge; }
std::string_view Gauge::name() const { return name_; }
std::string_view Gauge::help() const { return help_; }
const Labels& Gauge::labels() const { return labels_; }
void Gauge::AppendSamples(std::string* output) const {
  output->append(name_);
  AppendLabels(labels_, {}, {}, output);
  output->push_back(' ');
  AppendDouble(value(), output);
  output->push_back('\n');
}

Histogram::Histogram(std::string name, std::string help,
                     std::vector<double> buckets, Labels labels)
    : name_(std::move(name)),
      help_(std::move(help)),
      labels_(std::move(labels)),
      buckets_(std::move(buckets)) {
  std::sort(labels_.begin(), labels_.end());
  ValidateDescriptor(name_, labels_);
  if (std::any_of(labels_.begin(), labels_.end(),
                  [](const auto& label) { return label.first == "le"; })) {
    throw std::invalid_argument("histogram label 'le' is reserved");
  }
  if (!std::all_of(buckets_.begin(), buckets_.end(),
                   [](double value) { return std::isfinite(value); }) ||
      !std::is_sorted(buckets_.begin(), buckets_.end()) ||
      std::adjacent_find(buckets_.begin(), buckets_.end()) != buckets_.end()) {
    throw std::invalid_argument(
        "histogram buckets must be finite and increasing");
  }
  bucket_counts_ = std::make_unique<std::atomic<uint64_t>[]>(buckets_.size());
  for (size_t i = 0; i < buckets_.size(); ++i) bucket_counts_[i].store(0);
}

void Histogram::Observe(double value) noexcept {
  if (std::isnan(value)) return;
  const auto position =
      std::lower_bound(buckets_.begin(), buckets_.end(), value);
  if (position != buckets_.end()) {
    bucket_counts_[position - buckets_.begin()].fetch_add(
        1, std::memory_order_relaxed);
  }
  count_.fetch_add(1, std::memory_order_relaxed);
  sum_.fetch_add(value, std::memory_order_relaxed);
}
uint64_t Histogram::count() const noexcept {
  return count_.load(std::memory_order_relaxed);
}
Metric::Type Histogram::type() const { return Type::kHistogram; }
std::string_view Histogram::name() const { return name_; }
std::string_view Histogram::help() const { return help_; }
const Labels& Histogram::labels() const { return labels_; }
void Histogram::AppendSamples(std::string* output) const {
  uint64_t cumulative = 0;
  for (size_t i = 0; i < buckets_.size(); ++i) {
    cumulative += bucket_counts_[i].load(std::memory_order_relaxed);
    output->append(name_).append("_bucket");
    std::string boundary;
    AppendDouble(buckets_[i], &boundary);
    AppendLabels(labels_, "le", boundary, output);
    output->push_back(' ');
    output->append(std::to_string(cumulative)).push_back('\n');
  }
  output->append(name_).append("_bucket");
  AppendLabels(labels_, "le", "+Inf", output);
  output->push_back(' ');
  output->append(std::to_string(count())).push_back('\n');
  output->append(name_).append("_sum");
  AppendLabels(labels_, {}, {}, output);
  output->push_back(' ');
  AppendDouble(sum_.load(std::memory_order_relaxed), output);
  output->push_back('\n');
  output->append(name_).append("_count");
  AppendLabels(labels_, {}, {}, output);
  output->push_back(' ');
  output->append(std::to_string(count())).push_back('\n');
}

void Registry::AddMetric(std::shared_ptr<Metric> metric) {
  std::lock_guard lock(mutex_);
  for (const auto& existing : metrics_) {
    if (existing->name() != metric->name()) continue;
    if (existing->type() != metric->type() ||
        existing->help() != metric->help()) {
      throw std::invalid_argument("inconsistent metric family descriptor");
    }
    if (LabelKey(existing->labels()) == LabelKey(metric->labels())) {
      throw std::invalid_argument("duplicate metric label set");
    }
  }
  metrics_.push_back(std::move(metric));
}

std::shared_ptr<Counter> Registry::AddCounter(std::string name,
                                              std::string help, Labels labels) {
  auto metric = std::make_shared<Counter>(std::move(name), std::move(help),
                                          std::move(labels));
  AddMetric(metric);
  return metric;
}
std::shared_ptr<Gauge> Registry::AddGauge(std::string name, std::string help,
                                          Labels labels) {
  auto metric = std::make_shared<Gauge>(std::move(name), std::move(help),
                                        std::move(labels));
  AddMetric(metric);
  return metric;
}
std::shared_ptr<Histogram> Registry::AddHistogram(std::string name,
                                                  std::string help,
                                                  std::vector<double> buckets,
                                                  Labels labels) {
  auto metric = std::make_shared<Histogram>(
      std::move(name), std::move(help), std::move(buckets), std::move(labels));
  AddMetric(metric);
  return metric;
}

std::string Registry::Render() const {
  std::vector<std::shared_ptr<Metric>> snapshot;
  {
    std::lock_guard lock(mutex_);
    snapshot = metrics_;
  }
  std::stable_sort(
      snapshot.begin(), snapshot.end(),
      [](const auto& a, const auto& b) { return a->name() < b->name(); });
  std::string output;
  std::string_view family;
  for (const auto& metric : snapshot) {
    if (metric->name() != family) {
      family = metric->name();
      output.append("# HELP ").append(family).push_back(' ');
      AppendEscaped(metric->help(), &output, true);
      output.push_back('\n');
      output.append("# TYPE ").append(family).push_back(' ');
      output.append(TypeName(metric->type())).push_back('\n');
    }
    metric->AppendSamples(&output);
  }
  return output;
}

}  // namespace inferx::observe
