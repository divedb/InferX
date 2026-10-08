/// \file
/// \brief Runner-private fixed-capacity batch buffers: one device storage
///        for every flat input, a pinned host mirror, and exact-size views
///        per step. Addresses stay stable for graph capture; the padded
///        tails of the capacity buffers are never visible to the model.

#ifndef INFERX_MODELS_EXECUTION_BATCH_BUFFERS_H_
#define INFERX_MODELS_EXECUTION_BATCH_BUFFERS_H_

#include <optional>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"

namespace inferx::models::execution {

/// \brief One step's flat host-side packing, staged then uploaded.
struct PackedStep {
  std::vector<int32_t> tokens;
  std::vector<int32_t> positions;
  std::vector<int32_t> batch_indices;
  std::vector<int32_t> qo_indptr;   ///< [num_seqs + 1]
  std::vector<int32_t> kv_indptr;   ///< [num_seqs + 1]
  std::vector<int32_t> kv_indices;  ///< Flattened block table.
  std::vector<int32_t> last_page_len;
  std::vector<int32_t> logit_rows;  ///< Selected output rows only.
  std::optional<std::vector<int32_t>> recurrent_slots;
};

class BatchBuffers {
 public:
  /// \brief Allocates the device storage, the pinned host mirror, and the
  ///        host sample landing buffer. Fixed offsets keep graph pointers
  ///        stable; one pinned upload serves the whole step.
  Status Init(DeviceRuntime& runtime, DeviceId device, int max_tokens, int max_seqs,
              int num_kv_blocks, bool recurrent);

  /// \brief Exact-size device views for a packed step.
  struct Views {
    Tensor token_ids;
    Tensor positions;
    Tensor batch_indices;
    Tensor qo_indptr;
    Tensor kv_indptr;
    Tensor kv_indices;
    Tensor last_page_len;
    Tensor logit_rows;
    std::optional<Tensor> recurrent_indices;
  };

  /// \brief Stages `packed` into the pinned mirror and enqueues one
  ///        host-to-device copy on `stream`, then returns the views.
  StatusOr<Views> Upload(const PackedStep& packed, int num_seqs, Stream stream);

  /// \brief Pinned landing buffer for [max_seqs] sampled token ids.
  int32_t* host_samples() const { return host_samples_; }

  int max_tokens() const { return max_tokens_; }
  int max_seqs() const { return max_seqs_; }

 private:
  Tensor* input_view(int64_t offset_elements, int64_t count);

  DeviceRuntime* runtime_ = nullptr;
  DeviceId device_;
  int max_tokens_ = 0;
  int max_seqs_ = 0;
  int num_kv_blocks_ = 0;
  bool recurrent_ = false;
  std::optional<Tensor> storage_;
  // Fixed element offsets within storage_.
  int64_t token_ids_at_ = 0, positions_at_ = 0, batch_indices_at_ = 0;
  int64_t qo_indptr_at_ = 0, kv_indptr_at_ = 0, kv_indices_at_ = 0;
  int64_t last_page_len_at_ = 0, logit_rows_at_ = 0, recurrent_at_ = 0;
  int32_t* host_inputs_ = nullptr;
  int32_t* host_samples_ = nullptr;
};

}  // namespace inferx::models::execution

#endif  // INFERX_MODELS_EXECUTION_BATCH_BUFFERS_H_
