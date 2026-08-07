#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "inferx/core/status.h"

namespace inferx::server::model_registry {

enum class ModelState { kDiscovered, kLoading, kWarming, kReady, kDraining, kFailed };

struct ModelRecord {
  std::string id;
  std::string version;
  std::string alias;
  int64_t created = 0;
  ModelState state = ModelState::kDiscovered;
  bool supports_generation = true;
  bool supports_embeddings = false;
};

class Registry {
 public:
  Status Register(ModelRecord record);
  Status SetState(std::string_view id, std::string_view version,
                  ModelState state);
  StatusOr<ModelRecord> Resolve(std::string_view id_or_alias) const;
  std::vector<ModelRecord> ReadyModels() const;
  size_t size() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, ModelRecord> records_;
  std::unordered_map<std::string, std::string> aliases_;
};

}  // namespace inferx::server::model_registry
