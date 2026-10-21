/// \brief Incremental detokenizer service (docs/tokenizer_design.md §8.2).
///
/// Owns per-request decode state (token/text offsets, SGLang's DecodeStatus
/// pattern) and emits only text suffixes — never full-prefix duplicates and
/// never a partial UTF-8 character. Decode runs on a dedicated decode
/// worker, never on an I/O thread or the engine thread, over a tokenizer
/// instance that is never shared with prompt encoding. Append/Finish for
/// one request must not run concurrently; the future manager's
/// decode_running flag enforces that.
#ifndef INFERX_SERVER_DETOKENIZER_H_
#define INFERX_SERVER_DETOKENIZER_H_

#include <chrono>
#include <memory>
#include <string>
#include <vector>

#include <boost/asio.hpp>

#include "inferx/core/status.h"
#include "inferx/server/request_model.h"
#include "inferx/tokenizer/id.h"
#include "inferx/tokenizer/tokenizer.h"

namespace inferx {

/// \brief Incremental text suffix produced by Append/Finish; may be empty.
struct DecodeDelta {
  std::string text;
};

class Detokenizer {
 public:
  virtual ~Detokenizer() = default;

  /// \brief Appends generated tokens and returns the newly settled text
  /// suffix. The first Append for a request creates its decode state.
  /// Tokens are passed by value: an awaitable must not capture a span that
  /// may dangle.
  virtual boost::asio::awaitable<StatusOr<DecodeDelta>> Append(
      RequestId request_id, std::vector<TokenId> tokens,
      const DecodeOptions& options) = 0;

  /// \brief Flushes any held-back UTF-8/replacement bytes and releases the
  /// decode state. Returns the final text suffix (possibly empty).
  virtual boost::asio::awaitable<StatusOr<DecodeDelta>> Finish(
      RequestId request_id) = 0;

  /// \brief Drops the request's decode state immediately; idempotent.
  virtual void Cancel(RequestId request_id) = 0;

  /// \brief Stops admitting new work; queued jobs drain, then the decode
  /// worker exits. Call JoinUntil to observe completion.
  virtual void BeginDrain() = 0;

  /// \brief Completes once the decode worker has exited, or the deadline
  /// passes.
  virtual boost::asio::awaitable<Status> JoinUntil(
      std::chrono::steady_clock::time_point deadline) = 0;
};

/// \brief Creates the production detokenizer: one decode worker thread that
/// loads its own tokenizer instance from `tokenizer_path`. The worker
/// reports unready if the asset fails to load; Append/Finish then return
/// Unavailable.
std::shared_ptr<Detokenizer> CreateDetokenizer(const std::string& tokenizer_path);

/// \brief Same, over a caller-supplied tokenizer instance that the
/// detokenizer owns exclusively from here on (used by embedders and tests).
std::shared_ptr<Detokenizer> CreateDetokenizer(
    std::shared_ptr<Tokenizer> exclusive_tokenizer);

}  // namespace inferx

#endif  // INFERX_SERVER_DETOKENIZER_H_
