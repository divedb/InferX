/// \file
/// \brief Bounded waiting queue for requests (plan.md D3/D14).

#ifndef INFERX_ENGINE_REQUEST_QUEUE_H_
#define INFERX_ENGINE_REQUEST_QUEUE_H_

#include <cstddef>
#include <deque>

#include "inferx/core/status.h"
#include "inferx/engine/request.h"

namespace inferx {

/// \brief FIFO waiting queue with an explicit capacity.
///
/// The bound is the engine's admission line: Push fails with
/// ResourceExhausted -- the status the serving layer turns into 429/503 --
/// instead of letting latency grow without limit. Unbounded waiting queues
/// are a defect (plan.md principle 3). A priority policy slots in behind
/// this same interface when D3's aging/priority work lands; it is FCFS
/// only for now.
class RequestQueue {
 public:
  /// \brief Constructs a queue holding at most `capacity` requests.
  ///
  /// \param capacity Maximum queued requests; zero rejects everything.
  explicit RequestQueue(std::size_t capacity);

  /// \brief Enqueues a request.
  ///
  /// \param request The request to enqueue.
  /// \return        OK, or ResourceExhausted when the queue is full.
  Status Push(Request request);

  /// \brief Re-queues a request at the head, ahead of older arrivals.
  ///
  /// Used by the scheduler to put back a request it popped but could not
  /// admit; capacity checks are skipped because the queue just shrank.
  ///
  /// \param request The request to put back.
  void PushFront(Request request);

  /// \brief Dequeues the oldest request.
  ///
  /// \return The request, or NotFound when the queue is empty.
  StatusOr<Request> Pop();

  /// \brief Removes a queued request by id.
  ///
  /// \param request_id The request to remove.
  /// \return           True when the request was queued and is now removed.
  bool Erase(uint64_t request_id);

  /// \brief True when no requests are queued.
  bool empty() const { return requests_.empty(); }
  /// \brief Returns the number of queued requests.
  std::size_t size() const { return requests_.size(); }
  /// \brief Returns the maximum number of queued requests.
  std::size_t capacity() const { return capacity_; }

 private:
  std::deque<Request> requests_;
  std::size_t capacity_ = 0;
};

}  // namespace inferx

#endif  // INFERX_ENGINE_REQUEST_QUEUE_H_
