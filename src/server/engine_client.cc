#include "inferx/server/engine_client.h"

#include <atomic>
#include <memory>
#include <utility>

namespace inferx::server {

namespace {

template <class Stream, class Client>
struct HandleState {
  HandleState(std::weak_ptr<Client> client_in, EngineRequestId id_in,
              std::unique_ptr<Stream> events_in)
      : client(std::move(client_in)),
        id(std::move(id_in)),
        events(std::move(events_in)) {}

  std::weak_ptr<Client> client;
  EngineRequestId id;
  std::unique_ptr<Stream> events;  // Close() is thread-safe and idempotent
  std::atomic<bool> cancel_requested{false};
};

}  // namespace

// ---- EngineRequestHandle ----

struct EngineRequestHandle::State
    : HandleState<IEngineEventStream, IEngineClient> {
  using HandleState::HandleState;
};

EngineRequestHandle::EngineRequestHandle(
    std::weak_ptr<IEngineClient> client, EngineRequestId id,
    std::unique_ptr<IEngineEventStream> events)
    : state_(std::make_shared<State>(std::move(client), std::move(id),
                                     std::move(events))) {}

EngineRequestHandle::~EngineRequestHandle() {
  if (state_ && state_->events) {
    Cancel(CancelReason::kInternal);
  }
}

EngineRequestHandle::EngineRequestHandle(EngineRequestHandle&& other) noexcept
    : state_(std::move(other.state_)) {}

EngineRequestHandle& EngineRequestHandle::operator=(
    EngineRequestHandle&& other) noexcept {
  if (this != &other) {
    if (state_ && state_->events) {
      Cancel(CancelReason::kInternal);
    }
    state_ = std::move(other.state_);
  }
  return *this;
}

const EngineRequestId& EngineRequestHandle::id() const noexcept {
  static const EngineRequestId kEmpty;
  return state_ ? state_->id : kEmpty;
}

IEngineEventStream& EngineRequestHandle::events() const {
  return *state_->events;
}

bool EngineRequestHandle::valid() const noexcept {
  return state_ && state_->events != nullptr;
}

void EngineRequestHandle::Cancel(CancelReason reason) noexcept {
  if (!state_) return;
  if (state_->cancel_requested.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  if (state_->events) state_->events->Close();
  if (auto client = state_->client.lock()) {
    client->Abort(state_->id, reason);
  }
}

bool EngineRequestHandle::cancel_requested() const noexcept {
  return state_ && state_->cancel_requested.load(std::memory_order_acquire);
}

// ---- EngineEmbedHandle ----

struct EngineEmbedHandle::State
    : HandleState<IEngineEmbedStream, IEngineClient> {
  using HandleState::HandleState;
};

EngineEmbedHandle::EngineEmbedHandle(
    std::weak_ptr<IEngineClient> client, EngineRequestId id,
    std::unique_ptr<IEngineEmbedStream> events)
    : state_(std::make_shared<State>(std::move(client), std::move(id),
                                     std::move(events))) {}

EngineEmbedHandle::~EngineEmbedHandle() {
  if (state_ && state_->events) {
    Cancel(CancelReason::kInternal);
  }
}

EngineEmbedHandle::EngineEmbedHandle(EngineEmbedHandle&& other) noexcept
    : state_(std::move(other.state_)) {}

EngineEmbedHandle& EngineEmbedHandle::operator=(
    EngineEmbedHandle&& other) noexcept {
  if (this != &other) {
    if (state_ && state_->events) {
      Cancel(CancelReason::kInternal);
    }
    state_ = std::move(other.state_);
  }
  return *this;
}

const EngineRequestId& EngineEmbedHandle::id() const noexcept {
  static const EngineRequestId kEmpty;
  return state_ ? state_->id : kEmpty;
}

IEngineEmbedStream& EngineEmbedHandle::events() const {
  return *state_->events;
}

bool EngineEmbedHandle::valid() const noexcept {
  return state_ && state_->events != nullptr;
}

void EngineEmbedHandle::Cancel(CancelReason reason) noexcept {
  if (!state_) return;
  if (state_->cancel_requested.exchange(true, std::memory_order_acq_rel)) {
    return;
  }
  if (state_->events) state_->events->Close();
  if (auto client = state_->client.lock()) {
    client->Abort(state_->id, reason);
  }
}

bool EngineEmbedHandle::cancel_requested() const noexcept {
  return state_ && state_->cancel_requested.load(std::memory_order_acquire);
}

}  // namespace inferx::server
