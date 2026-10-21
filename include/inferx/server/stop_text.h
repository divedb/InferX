#ifndef INFERX_SERVER_STOP_TEXT_H_
#define INFERX_SERVER_STOP_TEXT_H_

#include <algorithm>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace inferx {

// Consumes settled decoded deltas (after TextDelta's UTF-8 handling).
// Retains any suffix that could become a stop string in a later delta.
class StopText {
 public:
  explicit StopText(std::vector<std::string> stops = {}) : stops_(std::move(stops)) {}

  std::string Append(std::string_view delta) {
    if (stopped_) return {};
    pending_.append(delta);
    std::size_t match = std::string::npos;
    for (const auto& stop : stops_) {
      if (!stop.empty()) match = std::min(match, pending_.find(stop));
    }
    if (match != std::string::npos) {
      stopped_ = true;
      auto text = pending_.substr(0, match);
      pending_.clear();
      return text;
    }
    std::size_t held = 0;
    for (const auto& stop : stops_) {
      for (std::size_t n = 1; n < stop.size() && n <= pending_.size(); ++n) {
        if (std::string_view(pending_).substr(pending_.size() - n) ==
            std::string_view(stop).substr(0, n)) {
          held = std::max(held, n);
        }
      }
    }
    auto text = pending_.substr(0, pending_.size() - held);
    pending_.erase(0, text.size());
    return text;
  }

  std::string Finish() {
    auto tail = std::move(pending_);
    pending_.clear();
    return tail;
  }
  bool stopped() const { return stopped_; }

 private:
  std::vector<std::string> stops_;
  std::string pending_;
  bool stopped_ = false;
};

}  // namespace inferx
#endif
