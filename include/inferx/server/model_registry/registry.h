#pragma once

#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "inferx/core/status.h"

namespace inferx::server::model_registry {

enum class ModelState {
  kDiscovered,
  kDownloading,
  kLoading,
  kWarming,
  kReady,
  kDraining,
  kUnloading,
  kUnloaded,
  kFailed
};

struct ModelRecord {
  std::string id;
  std::string version;
  std::string alias;
  int64_t created = 0;
  ModelState state = ModelState::kDiscovered;
  bool supports_generation = true;
  bool supports_embeddings = false;
  uint32_t context_limit = 0;
  uint32_t max_output_tokens = 0;
  uint32_t embedding_dimensions = 0;
  uint32_t embedding_max_batch_size = 1;
  std::unordered_set<std::string> embedding_encoding_formats{"float"};
  std::string tokenizer_revision;
  std::unordered_set<std::string> visible_tenants;
};

bool CanTransition(ModelState from, ModelState to);

class Registry {
 public:
  Status Register(ModelRecord record);
  Status SetState(std::string_view id, std::string_view version,
                  ModelState state);
  StatusOr<ModelRecord> Resolve(std::string_view id_or_alias) const;
  StatusOr<ModelRecord> Resolve(std::string_view id_or_alias,
                                std::string_view tenant) const;
  std::vector<ModelRecord> ReadyModels() const;
  std::vector<ModelRecord> ReadyModels(std::string_view tenant) const;
  std::vector<ModelRecord> Models(std::string_view tenant) const;
  size_t size() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, ModelRecord> records_;
  std::unordered_map<std::string, std::string> aliases_;
};

}  // namespace inferx::server::model_registry
