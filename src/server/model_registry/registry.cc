#include "inferx/server/model_registry/registry.h"

#include <utility>

namespace inferx::server::model_registry {
namespace {

std::string Key(std::string_view id, std::string_view version) {
  return std::string(id) + "@" + std::string(version);
}

}  // namespace

Status Registry::Register(ModelRecord record) {
  if (record.id.empty() || record.version.empty()) {
    return InvalidArgumentError("model ID and version are required");
  }
  std::lock_guard lock(mutex_);
  const std::string key = Key(record.id, record.version);
  if (records_.contains(key)) return FailedPreconditionError("model version exists");
  if (!record.alias.empty() && aliases_.contains(record.alias)) {
    return FailedPreconditionError("model alias exists: ", record.alias);
  }
  if (!record.alias.empty()) aliases_.emplace(record.alias, key);
  records_.emplace(key, std::move(record));
  return OkStatus();
}

Status Registry::SetState(std::string_view id, std::string_view version,
                          ModelState state) {
  std::lock_guard lock(mutex_);
  const auto it = records_.find(Key(id, version));
  if (it == records_.end()) return NotFoundError("model version not found");
  it->second.state = state;
  return OkStatus();
}

StatusOr<ModelRecord> Registry::Resolve(std::string_view id_or_alias) const {
  std::lock_guard lock(mutex_);
  std::string key(id_or_alias);
  if (const auto alias = aliases_.find(key); alias != aliases_.end()) key = alias->second;
  const auto it = records_.find(key);
  if (it == records_.end() || it->second.state != ModelState::kReady) {
    return NotFoundError("ready model not found: ", id_or_alias);
  }
  return it->second;
}

std::vector<ModelRecord> Registry::ReadyModels() const {
  std::lock_guard lock(mutex_);
  std::vector<ModelRecord> result;
  for (const auto& [key, record] : records_) {
    (void)key;
    if (record.state == ModelState::kReady) result.push_back(record);
  }
  return result;
}

size_t Registry::size() const {
  std::lock_guard lock(mutex_);
  return records_.size();
}

}  // namespace inferx::server::model_registry
