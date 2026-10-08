#include "inferx/models/execution/stream_waiter.h"

#include <chrono>
#include <thread>

#include "inferx/core/logging.h"

namespace inferx::models::execution {

void StreamWaiter::SetHealthCheck(std::function<Status()> check) {
  check_health_ = std::move(check);
  if (!check_health_ || event_.handle != nullptr || runtime_ == nullptr) return;

  auto event = runtime_->CreateEvent(false);
  if (event.ok()) {
    event_ = *event;
  } else {
    INFERX_LOG(WARNING) << "health polling disabled: " << event.status().message();
  }
}

Status StreamWaiter::Wait() {
  if (!check_health_ || event_.handle == nullptr) return runtime_->SynchronizeStream(*stream_);
  INFERX_RETURN_IF_ERROR(runtime_->RecordEvent(event_, *stream_));

  while (true) {
    INFERX_RETURN_IF_ERROR(check_health_());
    INFERX_ASSIGN_OR_RETURN(bool done, runtime_->QueryEvent(event_));
    if (done) return OkStatus();
    std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

void StreamWaiter::Reset() {
  if (runtime_ != nullptr && event_.handle != nullptr) (void)runtime_->DestroyEvent(event_);
  event_ = DeviceEvent{};
}

}  // namespace inferx::models::execution
