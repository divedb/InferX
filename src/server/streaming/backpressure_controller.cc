#include "inferx/server/streaming/backpressure_controller.h"

namespace inferx::server::streaming {

void BackpressureController::Observe(
    PushResult result, std::chrono::steady_clock::time_point now) {
  if (result == PushResult::kFull) {
    if (!first_full_) first_full_ = now;
  } else if (result == PushResult::kQueued) {
    first_full_.reset();
  }
}

bool BackpressureController::ShouldCancel(
    std::chrono::steady_clock::time_point now) const {
  return first_full_.has_value() && now - *first_full_ >= timeout_;
}

std::chrono::steady_clock::time_point BackpressureController::first_full() const {
  return first_full_.value_or(std::chrono::steady_clock::time_point{});
}

}  // namespace inferx::server::streaming
