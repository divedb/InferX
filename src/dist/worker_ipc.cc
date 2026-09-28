#include "inferx/dist/worker_ipc.h"

#include <chrono>
#include <memory>
#include <string>
#include <utility>

#include "absl/strings/str_cat.h"
#include "boost/interprocess/ipc/message_queue.hpp"

namespace inferx {
namespace dist {
namespace {

using boost::interprocess::message_queue;

constexpr uint32_t kQueueDepth = 256;

Status ToStatus(const std::exception& e, const char* what) {
  return InternalError("ipc ", what, ": ", e.what());
}

}  // namespace

struct MessageChannel::Impl {
  std::string name;
  std::unique_ptr<message_queue> queue;
  bool owned = false;  // The creator removes the kernel object on teardown.
};

IpcChannelNames MakeIpcChannelNames(uint64_t session) {
  const std::string prefix = absl::StrCat("inferx_dist_", session);
  return IpcChannelNames{prefix + "_commands", prefix + "_events"};
}

StatusOr<MessageChannel> MessageChannel::Create(const std::string& name) {
  // A crashed prior session may have left the kernel object behind; the
  // session id makes stale names foreign, and removing a name we are about
  // to create is harmless.
  message_queue::remove(name.c_str());
  try {
    auto impl = std::make_unique<Impl>();
    impl->name = name;
    impl->owned = true;
    impl->queue = std::make_unique<message_queue>(
        boost::interprocess::create_only, name.c_str(), kQueueDepth,
        sizeof(WireMessage));
    return MessageChannel(std::move(impl));
  } catch (const std::exception& e) {
    return ToStatus(e, "create");
  }
}

StatusOr<MessageChannel> MessageChannel::Open(const std::string& name) {
  try {
    auto impl = std::make_unique<Impl>();
    impl->name = name;
    impl->queue = std::make_unique<message_queue>(
        boost::interprocess::open_only, name.c_str());
    return MessageChannel(std::move(impl));
  } catch (const std::exception& e) {
    return ToStatus(e, "open");
  }
}

Status MessageChannel::Send(const WireMessage& message) {
  try {
    impl_->queue->send(&message, sizeof(WireMessage), 0);
    return OkStatus();
  } catch (const std::exception& e) {
    return ToStatus(e, "send");
  }
}

bool MessageChannel::TryReceive(uint32_t timeout_ms, WireMessage* out) {
  const auto deadline =
      std::chrono::system_clock::now() + std::chrono::milliseconds(timeout_ms);
  try {
    unsigned int priority = 0;
    message_queue::size_type received = 0;
    if (impl_->queue->timed_receive(out, sizeof(WireMessage), received, priority,
                                    deadline)) {
      return received == sizeof(WireMessage);
    }
    return false;
  } catch (const std::exception&) {
    return false;
  }
}

void MessageChannel::Remove() {
  if (impl_ && impl_->owned) {
    message_queue::remove(impl_->name.c_str());
    impl_->owned = false;
  }
}

}  // namespace inferx::dist
}  // namespace inferx
