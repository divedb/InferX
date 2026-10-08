#include "inferx/models/execution/batch_buffers.h"

#include <cstring>

namespace inferx::models::execution {

Status BatchBuffers::Init(DeviceRuntime& runtime, DeviceId device, int max_tokens,
                          int max_seqs, int num_kv_blocks, bool recurrent) {
  runtime_ = &runtime;
  device_ = device;
  max_tokens_ = max_tokens;
  max_seqs_ = max_seqs;
  num_kv_blocks_ = num_kv_blocks;
  recurrent_ = recurrent;

  // 3 * T token-major arrays, 5 * S + 2 sequence-major arrays, and the
  // block table. Fixed offsets keep every graph pointer stable.
  INFERX_ASSIGN_OR_RETURN(
      storage_, Tensor::Empty(DataType::kInt32,
                              Shape({3 * int64_t(max_tokens) + 5 * int64_t(max_seqs) + 2 +
                                     int64_t(num_kv_blocks)}),
                              device));
  int64_t offset = 0;
  token_ids_at_ = offset;
  offset += max_tokens;
  positions_at_ = offset;
  offset += max_tokens;
  batch_indices_at_ = offset;
  offset += max_tokens;
  qo_indptr_at_ = offset;
  offset += max_seqs + 1;
  kv_indptr_at_ = offset;
  offset += max_seqs + 1;
  kv_indices_at_ = offset;
  offset += num_kv_blocks;
  last_page_len_at_ = offset;
  offset += max_seqs;
  logit_rows_at_ = offset;
  offset += max_seqs;
  recurrent_at_ = offset;  // + max_seqs when recurrent

  if (device.IsCuda()) {
    INFERX_ASSIGN_OR_RETURN(void* inputs, runtime.AllocatePinnedHost(storage_->NBytes()));
    host_inputs_ = static_cast<int32_t*>(inputs);
    std::memset(host_inputs_, 0, storage_->NBytes());
    INFERX_ASSIGN_OR_RETURN(void* host,
                            runtime.AllocatePinnedHost(max_seqs * sizeof(int32_t)));
    host_samples_ = static_cast<int32_t*>(host);
  }
  return OkStatus();
}

Tensor* BatchBuffers::input_view(int64_t offset_elements, int64_t count) {
  auto view = storage_->Slice(offset_elements, offset_elements + count);
  if (!view.ok()) return nullptr;
  return &*view;
}

StatusOr<BatchBuffers::Views> BatchBuffers::Upload(const PackedStep& packed, int num_seqs,
                                                   Stream stream) {
  const int num_tokens = static_cast<int>(packed.tokens.size());
  const int num_blocks = static_cast<int>(packed.kv_indices.size());
  const int num_rows = static_cast<int>(packed.logit_rows.size());
  if (num_tokens > max_tokens_ || num_seqs > max_seqs_ || num_blocks > num_kv_blocks_ ||
      num_rows > max_seqs_) {
    return InvalidArgumentError("packed step exceeds batch-buffer capacity");
  }
  auto stage = [&](int64_t at, const std::vector<int32_t>& values) {
    if (host_inputs_ != nullptr) {
      std::memcpy(host_inputs_ + at, values.data(), values.size() * sizeof(int32_t));
      return OkStatus();
    }
    // CPU device: tensors are host-visible; copy synchronously.
    auto view = storage_->Slice(at, at + static_cast<int64_t>(values.size()));
    if (!view.ok()) return view.status();
    return runtime_->Copy(view->Data(), values.data(), values.size() * sizeof(int32_t),
                          CopyKind::kHostToDevice);
  };
  INFERX_RETURN_IF_ERROR(stage(token_ids_at_, packed.tokens));
  INFERX_RETURN_IF_ERROR(stage(positions_at_, packed.positions));
  INFERX_RETURN_IF_ERROR(stage(batch_indices_at_, packed.batch_indices));
  INFERX_RETURN_IF_ERROR(stage(qo_indptr_at_, packed.qo_indptr));
  INFERX_RETURN_IF_ERROR(stage(kv_indptr_at_, packed.kv_indptr));
  INFERX_RETURN_IF_ERROR(stage(kv_indices_at_, packed.kv_indices));
  INFERX_RETURN_IF_ERROR(stage(last_page_len_at_, packed.last_page_len));
  INFERX_RETURN_IF_ERROR(stage(logit_rows_at_, packed.logit_rows));
  if (packed.recurrent_slots.has_value()) {
    INFERX_RETURN_IF_ERROR(stage(recurrent_at_, *packed.recurrent_slots));
  }
  if (host_inputs_ != nullptr) {
    INFERX_RETURN_IF_ERROR(runtime_->CopyAsync(storage_->Data(), host_inputs_,
                                               storage_->NBytes(), CopyKind::kHostToDevice,
                                               stream));
  }

  auto take = [&](int64_t at, int64_t count) -> StatusOr<Tensor> {
    return storage_->Slice(at, at + count);
  };
  INFERX_ASSIGN_OR_RETURN(auto token_ids, take(token_ids_at_, num_tokens));
  INFERX_ASSIGN_OR_RETURN(auto positions, take(positions_at_, num_tokens));
  INFERX_ASSIGN_OR_RETURN(auto batch_indices, take(batch_indices_at_, num_tokens));
  INFERX_ASSIGN_OR_RETURN(auto qo_indptr, take(qo_indptr_at_, num_seqs + 1));
  INFERX_ASSIGN_OR_RETURN(auto kv_indptr, take(kv_indptr_at_, num_seqs + 1));
  INFERX_ASSIGN_OR_RETURN(auto kv_indices, take(kv_indices_at_, num_blocks));
  INFERX_ASSIGN_OR_RETURN(auto last_page_len, take(last_page_len_at_, num_seqs));
  INFERX_ASSIGN_OR_RETURN(auto logit_rows, take(logit_rows_at_, num_rows));
  std::optional<Tensor> recurrent;
  if (recurrent_) {
    INFERX_ASSIGN_OR_RETURN(recurrent, take(recurrent_at_, num_seqs));
  }
  return Views{std::move(token_ids),     std::move(positions),    std::move(batch_indices),
               std::move(qo_indptr),     std::move(kv_indptr),    std::move(kv_indices),
               std::move(last_page_len), std::move(logit_rows),   std::move(recurrent)};
}

}  // namespace inferx::models::execution
