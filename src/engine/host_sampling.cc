#include "host_sampling.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace inferx::engine {

int32_t HostSampleRow(float* row, int64_t vocab,
                      const model::ForwardBatch& batch, size_t i,
                      SampledLogprob* logprob_out) {
  using FB = model::ForwardBatch;
  const auto at_f = [&](const std::vector<float>& v, float fallback) {
    return i < v.size() ? v[i] : fallback;
  };

  const float presence = at_f(batch.presence_penalty, 0.0f);
  const float frequency = at_f(batch.frequency_penalty, 0.0f);
  const float repetition = at_f(batch.repetition_penalty, 1.0f);
  const size_t hist_cap = static_cast<size_t>(FB::kPenaltyHistoryCap);
  if ((presence != 0.0f || frequency != 0.0f || repetition != 1.0f) &&
      batch.penalty_history_ids.size() >= (i + 1) * hist_cap) {
    for (size_t j = 0; j < hist_cap; ++j) {
      const int32_t id = batch.penalty_history_ids[i * hist_cap + j];
      if (id < 0 || static_cast<int64_t>(id) >= vocab) continue;
      float& value = row[id];
      if (repetition != 1.0f) {
        value = value > 0.0f ? value / repetition : value * repetition;
      }
      value -= presence;
      value -= frequency * static_cast<float>(
                               batch.penalty_history_counts[i * hist_cap + j]);
    }
  }

  const size_t mask_cap = static_cast<size_t>(FB::kMaskCap);
  if (batch.mask_token_ids.size() >= (i + 1) * mask_cap) {
    for (size_t j = 0; j < mask_cap; ++j) {
      const int32_t id = batch.mask_token_ids[i * mask_cap + j];
      if (id >= 0 && static_cast<int64_t>(id) < vocab) {
        row[id] = -std::numeric_limits<float>::infinity();
      }
    }
  }

  const float temperature = at_f(batch.temperature, 0.0f);
  int32_t chosen = 0;
  if (temperature <= 0.0f) {
    chosen = static_cast<int32_t>(std::max_element(row, row + vocab) - row);
  } else {
    const float top_p = at_f(batch.top_p, 1.0f);
    const int32_t top_k = i < batch.top_k.size() ? batch.top_k[i] : 0;
    const float min_p = at_f(batch.min_p, 0.0f);
    const uint64_t seed = i < batch.seeds.size() ? batch.seeds[i] : 0;

    const float inv_t = 1.0f / temperature;
    float max_logit = -std::numeric_limits<float>::infinity();
    for (int64_t v = 0; v < vocab; ++v) {
      max_logit = std::max(max_logit, row[v] * inv_t);
    }
    std::vector<std::pair<float, int32_t>> probs;
    probs.reserve(static_cast<size_t>(vocab));
    double total = 0.0;
    for (int64_t v = 0; v < vocab; ++v) {
      const float p = std::exp(row[v] * inv_t - max_logit);
      total += p;
      probs.emplace_back(p, static_cast<int32_t>(v));
    }

    std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) {
      return a.first != b.first ? a.first > b.first : a.second < b.second;
    });

    size_t keep = probs.size();
    if (top_k > 0) keep = std::min<size_t>(keep, static_cast<size_t>(top_k));
    if (min_p > 0.0f) {
      const double floor_prob = static_cast<double>(min_p) * probs[0].first;
      size_t n = 0;
      while (n < keep && probs[n].first >= floor_prob) ++n;
      keep = std::max<size_t>(n, 1);
    }
    double mass = 0.0;
    if (top_p < 1.0f) {
      size_t n = 0;
      while (n < keep) {
        mass += probs[n].first / total;
        ++n;
        if (mass >= top_p) break;
      }
      keep = std::max<size_t>(n, 1);
    } else {
      for (size_t n = 0; n < keep; ++n) mass += probs[n].first / total;
    }

    std::mt19937_64 rng(seed);
    const double target =
        std::uniform_real_distribution<double>(0.0, mass)(rng);
    double running = 0.0;
    chosen = probs[keep - 1].second;
    for (size_t n = 0; n < keep; ++n) {
      running += probs[n].first / total;
      if (running >= target) {
        chosen = probs[n].second;
        break;
      }
    }
  }

  const int32_t logprob_k =
      i < batch.logprobs_k.size() ? batch.logprobs_k[i] : -1;
  if (logprob_out != nullptr && logprob_k >= 0) {
    float raw_max = -std::numeric_limits<float>::infinity();
    for (int64_t v = 0; v < vocab; ++v) raw_max = std::max(raw_max, row[v]);
    double raw_total = 0.0;
    for (int64_t v = 0; v < vocab; ++v) {
      raw_total += std::exp(row[v] - raw_max);
    }
    const float log_z = static_cast<float>(std::log(raw_total)) + raw_max;

    logprob_out->present = true;
    logprob_out->logprob = row[chosen] - log_z;

    const size_t want =
        std::min<size_t>(static_cast<size_t>(logprob_k),
                         static_cast<size_t>(FB::kMaxTopLogprobs));
    if (want > 0) {
      std::vector<int32_t> order(static_cast<size_t>(vocab));
      std::iota(order.begin(), order.end(), 0);
      std::partial_sort(order.begin(), order.begin() + want, order.end(),
                        [&](int32_t a, int32_t b) {
                          return row[a] != row[b] ? row[a] > row[b] : a < b;
                        });
      for (size_t j = 0; j < want; ++j) {
        logprob_out->top.emplace_back(order[j], row[order[j]] - log_z);
      }
    }
  }

  return chosen;
}

}  // namespace inferx::engine
