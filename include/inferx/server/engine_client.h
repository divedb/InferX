/// \brief Transport-neutral engine client for the HTTP layer
/// (docs/inference_server_design.md §7.5, reconciled to InferX).
///
/// The HTTP layer never touches the scheduler, gateway internals, or GPU
/// state; it submits already-tokenized completions through this interface
/// and pulls events one at a time (the doc's backpressure rule: no
/// unbounded SSE queue — the handler awaits each socket write before
/// pulling the next event). The production adapter wraps EngineGateway
/// (serve.cc); tests inject a deterministic fake so the whole HTTP surface
/// is testable without the engine.
#ifndef INFERX_SERVER_ENGINE_CLIENT_H_
#define INFERX_SERVER_ENGINE_CLIENT_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <boost/asio.hpp>

#include "inferx/core/status.h"
#include "inferx/engine/request.h"
#include "inferx/sampling/sampling_params.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

namespace net = boost::asio;

/// \brief One generation submission after prompt preparation: token ids
/// plus validated sampling. The engine remains authoritative for admission.
struct EngineCompletion {
  std::vector<TokenId> prompt;
  sampling::SamplingParams params;
};

/// \brief One engine event, pulled via IEngineEventStream::Next. `seq` is
/// strictly increasing within a stream; duplicates or gaps are protocol
/// errors the handler must not concatenate across.
struct EngineEvent {
  enum class Kind : std::uint8_t { kDelta, kFinished, kError };
  Kind kind = Kind::kDelta;
  std::uint64_t seq = 0;
  std::string text;  ///< kDelta: incremental text suffix, valid UTF-8.

  FinishReason reason = FinishReason::kStopped;  ///< kFinished.
  int prompt_tokens = 0;                         ///< kFinished.
  int completion_tokens = 0;                     ///< kFinished.

  bool admission = false;  ///< kError: true when the engine rejected the
                           ///< request at admission (503-class), false for
                           ///< internal errors (500-class).
  Status error;            ///< kError detail.
};

/// \brief Pull-based per-request event stream. Returning an error Status
/// means the stream itself broke (cancelled, closed, transport failure);
/// request-level failures arrive as kError events. Close() is idempotent.
class IEngineEventStream {
 public:
  virtual ~IEngineEventStream() = default;
  virtual net::awaitable<StatusOr<EngineEvent>> Next() = 0;
  virtual void Close() = 0;
};

struct EngineSubmission {
  std::string request_id;  ///< Engine-visible id for Abort correlation.
  std::unique_ptr<IEngineEventStream> events;
};

class IEngineClient {
 public:
  virtual ~IEngineClient() = default;

  /// \brief Submits a completion. A non-OK Status is an admission-class
  /// rejection (map to 429/503); acceptance means the engine queued the
  /// request, not that generation finished.
  virtual StatusOr<EngineSubmission> Submit(EngineCompletion request) = 0;

  /// \brief Idempotent abort; safe from any thread. A pending Next() wakes
  /// with a stream error afterwards.
  virtual void Abort(const std::string& request_id) = 0;
};

}  // namespace inferx

#endif  // INFERX_SERVER_ENGINE_CLIENT_H_
