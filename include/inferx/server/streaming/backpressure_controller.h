#pragma once

#include <chrono>
#include <optional>

#include "inferx/server/streaming/event_buffer.h"

namespace inferx::server::streaming {

class BackpressureController {
 public:
  explicit BackpressureController(std::chrono::milliseconds timeout)
      : timeout_(timeout) {}

  void Observe(PushResult result,
               std::chrono::steady_clock::time_point now =
                   std::chrono::steady_clock::now());
  bool ShouldCancel(std::chrono::steady_clock::time_point now =
                        std::chrono::steady_clock::now()) const;
  bool full() const { return first_full_.has_value(); }
  std::chrono::steady_clock::time_point first_full() const;

 private:
  std::chrono::milliseconds timeout_;
  std::optional<std::chrono::steady_clock::time_point> first_full_;
};

}  // namespace inferx::server::streaming
