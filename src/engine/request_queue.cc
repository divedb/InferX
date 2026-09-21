#include "inferx/engine/request_queue.h"

#include <utility>

namespace inferx {

RequestQueue::RequestQueue(std::size_t capacity) : capacity_(capacity) {}

Status RequestQueue::Push(Request request) {
  if (requests_.size() >= capacity_) {
    return ResourceExhaustedError("waiting queue is full (capacity ",
                                  capacity_, ")");
  }
  requests_.push_back(std::move(request));
  return OkStatus();
}

StatusOr<Request> RequestQueue::Pop() {
  if (requests_.empty()) {
    return NotFoundError("waiting queue is empty");
  }
  Request front = std::move(requests_.front());
  requests_.pop_front();
  return front;
}

void RequestQueue::PushFront(Request request) {
  requests_.push_front(std::move(request));
}

bool RequestQueue::Erase(uint64_t request_id) {
  for (auto it = requests_.begin(); it != requests_.end(); ++it) {
    if (it->id() == request_id) {
      requests_.erase(it);
      return true;
    }
  }
  return false;
}

}  // namespace inferx
