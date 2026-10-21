/// \brief Prompt preparation contract (docs/tokenizer_design.md §7.3).
///
/// A PromptProcessor runs on a prompt worker with exclusive use of its
/// tokenizer/template assets for the duration of a Prepare call. It has no
/// access to the request registry, the scheduler, or response sockets.
#ifndef INFERX_SERVER_PROMPT_PROCESSOR_H_
#define INFERX_SERVER_PROMPT_PROCESSOR_H_

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/server/request_model.h"
#include "inferx/tokenizer/id.h"

namespace inferx {

struct PromptLimits {
  std::size_t max_prompt_bytes = 8U << 20;
  std::size_t max_prompt_tokens = 0;  ///< 0 means "use model context limit".
  std::size_t max_media_bytes = 32U << 20;
  std::size_t max_media_items = 16;
};

struct PrepareContext {
  RequestId request_id = 0;
  Clock::time_point deadline{};  ///< Zero means "no explicit deadline".
  PromptLimits limits;
};

/// Processed media is already model-specific. It is not the raw API payload.
/// `data` is a byte buffer at the wire boundary; never a shared_ptr.
struct ProcessedMedia {
  Modality modality = Modality::kImage;
  std::string mime_type;
  std::string dtype;
  std::vector<std::int64_t> shape;
  std::vector<std::byte> data;
  std::vector<std::int32_t> token_positions;  ///< Placeholder token positions.
};

struct PreparedPrompt {
  std::vector<TokenId> token_ids;
  int prompt_tokens = 0;
  std::uint64_t tokenizer_fingerprint = 0;
  std::vector<ProcessedMedia> processed_media;  ///< Empty for text/token-only.
};

class PromptProcessor {
 public:
  virtual ~PromptProcessor() = default;

  /// Runs on a prompt worker. The processor has exclusive use of its
  /// tokenizer and template assets for the duration of this call.
  virtual StatusOr<PreparedPrompt> Prepare(const PromptInput& input,
                                           const PrepareContext& context) = 0;
};

}  // namespace inferx

#endif  // INFERX_SERVER_PROMPT_PROCESSOR_H_
