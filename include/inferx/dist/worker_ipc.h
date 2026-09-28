/// \file
/// \brief Controller <-> worker IPC: a fixed-size wire protocol over
///        Boost.InterProcess message queues.
///
/// Message queues are kernel-persistent and cross-process -- the
/// communication primitive already vendored in third_party/boost. Every
/// wire unit is the same size (the union of all messages), padded, so one
/// queue carries all kinds with a kind tag in the header. V1 protocol:
/// generate / shutdown downward, ready / token / finished upward.

#ifndef INFERX_DIST_WORKER_IPC_H_
#define INFERX_DIST_WORKER_IPC_H_

#include <cstdint>
#include <string>

#include "inferx/core/status.h"
#include "inferx/tokenizer/id.h"

namespace inferx {
namespace dist {

/// \brief Prompts ride inline in one message; longer prompts are rejected
///        until streaming segmentation lands.
inline constexpr uint32_t kMaxIpcPromptTokens = 512;

enum class MessageKind : uint32_t {
  kGenerate = 0,  ///< Controller -> worker: one generation request.
  kShutdown = 1,  ///< Controller -> worker: exit the loop.
  kReady = 2,     ///< Worker -> controller: engine loaded, rank/world attached.
  kToken = 3,     ///< Worker -> controller: one generated token.
  kFinished = 4,  ///< Worker -> controller: terminal state for a request.
  kFailed = 5,    ///< Worker -> controller: worker-level failure.
};

enum class FinishKind : uint8_t {
  kStopped = 0,
  kLengthCapped = 1,
  kAborted = 2,
  kError = 3,
};

/// \brief The fixed-size wire unit; one queue, kind-tagged.
struct WireMessage {
  MessageKind kind;
  union {
    struct {
      uint64_t request_id;
      uint32_t num_prompt;
      uint32_t max_tokens;
      TokenId prompt[kMaxIpcPromptTokens];
    } generate;
    struct {
      uint64_t request_id;
    } shutdown;
    struct {
      int32_t rank;
      int32_t world;
    } ready;
    struct {
      uint64_t request_id;
      TokenId token;
    } token;
    struct {
      uint64_t request_id;
      FinishKind finish;
    } finished;
    struct {
      char text[224];
    } failed;
  };
};

/// \brief Kernel-object names for one worker session.
struct IpcChannelNames {
  std::string commands;  ///< Controller -> worker.
  std::string events;    ///< Worker -> controller.
};

/// \brief Session-unique channel names (message queues persist in the
///        kernel until removed, so names must not collide across runs).
IpcChannelNames MakeIpcChannelNames(uint64_t session);

/// \brief One direction of the channel pair. The controller Creates (and
///        Removes on teardown); the worker Opens the same names.
class MessageChannel {
 public:
  static StatusOr<MessageChannel> Create(const std::string& name);
  static StatusOr<MessageChannel> Open(const std::string& name);

  MessageChannel() = default;
  MessageChannel(MessageChannel&&) = default;
  MessageChannel& operator=(MessageChannel&&) = default;

  /// \brief Enqueues one message (blocks if the queue is full).
  Status Send(const WireMessage& message);

  /// \brief Receives one message; waits up to `timeout_ms`.
  bool TryReceive(uint32_t timeout_ms, WireMessage* out);

  /// \brief Removes the kernel object (controller teardown only).
  void Remove();

 private:
  struct Impl;
  explicit MessageChannel(std::shared_ptr<Impl> impl) : impl_(std::move(impl)) {}
  std::shared_ptr<Impl> impl_;
};

}  // namespace inferx::dist
}  // namespace inferx

#endif  // INFERX_DIST_WORKER_IPC_H_
