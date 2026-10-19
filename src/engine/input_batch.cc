#include "inferx/models/input_batch.h"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

namespace inferx {
namespace models {

// NOTE: Status construction below assumes free helpers InvalidArgument(),
// AlreadyExists(), NotFound(), ResourceExhausted() and OkStatus() in
// inferx/core/status.h. Adjust to the project's actual factory names.

InputBatch::InputBatch(int max_num_reqs, int max_model_len)
    : max_num_reqs_(max_num_reqs),
      max_model_len_(max_model_len),
      token_ids_(static_cast<size_t>(max_num_reqs) * max_model_len),
      num_tokens_(max_num_reqs, 0),
      num_prompt_tokens_(max_num_reqs, 0),
      num_computed_(max_num_reqs, 0),
      params_(max_num_reqs),
      rng_offset_(max_num_reqs, 0) {
  row_to_id_.reserve(max_num_reqs);
  id_to_row_.reserve(max_num_reqs);
}

Status InputBatch::Add(const NewRequest& req) {
  if (id_to_row_.count(req.id) != 0) {
    return AlreadyExists("request " + std::to_string(req.id) + " is already in the batch");
  }
  if (static_cast<int>(req.prompt_tokens.size()) > max_model_len_) {
    return InvalidArgument("prompt of request " + std::to_string(req.id) +
                           " exceeds max_model_len");
  }

  int row;
  if (!free_rows_.empty()) {
    auto it = std::min_element(free_rows_.begin(), free_rows_.end());
    row = *it;
    free_rows_.erase(it);
  } else if (NumRows() < max_num_reqs_) {
    row = NumRows();
    row_to_id_.push_back(kFreeRow);
  } else {
    return ResourceExhausted("input batch is full");
  }

  const int n = static_cast<int>(req.prompt_tokens.size());
  if (n > 0) {
    std::memcpy(token_ids_.data() + static_cast<int64_t>(row) * max_model_len_,
                req.prompt_tokens.data(), n * sizeof(TokenId));
  }
  row_to_id_[row] = req.id;
  id_to_row_[req.id] = row;
  num_tokens_[row] = n;
  num_prompt_tokens_[row] = n;
  num_computed_[row] = req.num_computed;
  params_[row] = req.params;
  rng_offset_[row] = 0;
  return OkStatus();
}

void InputBatch::Remove(RequestId id) {
  auto it = id_to_row_.find(id);
  if (it == id_to_row_.end()) return;
  const int row = it->second;
  id_to_row_.erase(it);
  row_to_id_[row] = kFreeRow;
  num_tokens_[row] = 0;
  num_prompt_tokens_[row] = 0;
  num_computed_[row] = 0;
  rng_offset_[row] = 0;
  free_rows_.push_back(row);
}

void InputBatch::Condense() {
  if (free_rows_.empty()) return;

  std::vector<int> holes = std::move(free_rows_);
  free_rows_.clear();
  std::sort(holes.begin(), holes.end());

  // Fill the lowest holes with the highest live rows until they meet.
  int last = NumRows() - 1;
  for (int hole : holes) {
    while (last >= 0 && row_to_id_[last] == kFreeRow) --last;
    if (last < hole) break;

    const RequestId id = row_to_id_[last];
    const int n = num_tokens_[last];
    std::memcpy(token_ids_.data() + static_cast<int64_t>(hole) * max_model_len_,
                token_ids_.data() + static_cast<int64_t>(last) * max_model_len_,
                n * sizeof(TokenId));
    num_tokens_[hole] = n;
    num_prompt_tokens_[hole] = num_prompt_tokens_[last];
    num_computed_[hole] = num_computed_[last];
    params_[hole] = params_[last];
    rng_offset_[hole] = rng_offset_[last];

    row_to_id_[hole] = id;
    id_to_row_[id] = hole;
    row_to_id_[last] = kFreeRow;
    --last;
  }

  // After the moves, live rows are exactly [0, NumReqs()).
  row_to_id_.resize(id_to_row_.size());
}

Status InputBatch::AppendToken(RequestId id, TokenId token) {
  auto row = RowOf(id);
  if (!row) return NotFound("request " + std::to_string(id) + " is not in the batch");
  if (num_tokens_[*row] >= max_model_len_) {
    return ResourceExhausted("request " + std::to_string(id) + " reached max_model_len");
  }
  token_ids_[static_cast<int64_t>(*row) * max_model_len_ + num_tokens_[*row]] = token;
  ++num_tokens_[*row];
  return OkStatus();
}

Status InputBatch::SetNumComputed(RequestId id, int num_computed) {
  auto row = RowOf(id);
  if (!row) return NotFound("request " + std::to_string(id) + " is not in the batch");
  num_computed_[*row] = num_computed;
  return OkStatus();
}

std::optional<int> InputBatch::RowOf(RequestId id) const {
  auto it = id_to_row_.find(id);
  if (it == id_to_row_.end()) return std::nullopt;
  return it->second;
}

}  // namespace models
}  // namespace inferx