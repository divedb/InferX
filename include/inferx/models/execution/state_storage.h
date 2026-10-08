/// \file
/// \brief Runner-private persistent state storage: the paged-KV and
///        recurrent pools behind the model's declared requirements, plus
///        the per-layer bindings handed to Forward. Allocation policy
///        stays with the scheduler; this type only owns storage and
///        stream-ordered lifecycle operations.

#ifndef INFERX_MODELS_EXECUTION_STATE_STORAGE_H_
#define INFERX_MODELS_EXECUTION_STATE_STORAGE_H_

#include <memory>
#include <vector>

#include "inferx/cache/kv_block_pool.h"
#include "inferx/cache/recurrent_state_pool.h"
#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/models/requirements.h"

namespace inferx::models::execution {

class StateStorage {
 public:
  /// \brief Allocates the pools for the model's declared per-layer state.
  Status Init(const std::vector<LayerStateSpec>& specs, int num_kv_blocks, int block_size,
              int max_slots, DeviceId device);

  KvBlockPool* kv() { return kv_.get(); }
  const KvBlockPool* kv() const { return kv_.get(); }
  const RecurrentStatePool* recurrent() const { return recurrent_.get(); }
  RecurrentStatePool* mutable_recurrent() { return recurrent_.get(); }
  bool has_recurrent() const { return recurrent_ != nullptr; }

  /// \brief The per-layer bindings valid for every step of this model.
  const ModelState& bindings() const { return bindings_; }
  ModelState& bindings() { return bindings_; }

 private:
  std::unique_ptr<KvBlockPool> kv_;
  std::unique_ptr<RecurrentStatePool> recurrent_;
  ModelState bindings_;
};

}  // namespace inferx::models::execution

#endif  // INFERX_MODELS_EXECUTION_STATE_STORAGE_H_
