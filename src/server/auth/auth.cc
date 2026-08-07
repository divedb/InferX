#include "inferx/server/auth/api_key_store.h"
#include "inferx/server/auth/rbac.h"

namespace inferx::server::auth {

ApiKeyStore::ApiKeyStore() : entries_(std::make_shared<const Entries>()) {}

bool HasScope(const Principal& principal, std::string_view scope) {
  return principal.scopes.find(std::string(scope)) != principal.scopes.end();
}

Status ApiKeyStore::AddHash(std::string hash_hex, Principal principal) {
  INFERX_RETURN_IF_ERROR(Validate(hash_hex, principal));
  std::lock_guard lock(writer_mutex_);
  auto updated = std::make_shared<Entries>(*entries_.load());
  if (updated->contains(hash_hex)) {
    return FailedPreconditionError("API-key hash already exists");
  }
  updated->emplace(std::move(hash_hex), std::move(principal));
  entries_.store(std::move(updated));
  return OkStatus();
}

Status ApiKeyStore::RemoveHash(std::string_view hash_hex) {
  std::lock_guard lock(writer_mutex_);
  auto updated = std::make_shared<Entries>(*entries_.load());
  updated->erase(std::string(hash_hex));
  entries_.store(std::move(updated));
  return OkStatus();
}

Status ApiKeyStore::ReplaceAll(std::vector<ApiKeyRecord> records) {
  auto updated = std::make_shared<Entries>();
  updated->reserve(records.size());
  for (auto& record : records) {
    INFERX_RETURN_IF_ERROR(Validate(record.hash_hex, record.principal));
    const auto [unused, inserted] = updated->emplace(
        std::move(record.hash_hex), std::move(record.principal));
    if (!inserted) {
      return InvalidArgumentError("duplicate API-key hash in snapshot");
    }
  }

  std::lock_guard lock(writer_mutex_);
  entries_.store(std::move(updated));
  return OkStatus();
}

StatusOr<Principal> ApiKeyStore::LookupHash(std::string_view hash_hex) const {
  const std::shared_ptr<const Entries> snapshot = entries_.load();
  const auto it = snapshot->find(std::string(hash_hex));
  if (it == snapshot->end()) {
    return absl::UnauthenticatedError("invalid API key");
  }
  return it->second;
}

size_t ApiKeyStore::size() const { return entries_.load()->size(); }

Status ApiKeyStore::Validate(std::string_view hash_hex,
                             const Principal& principal) {
  if (hash_hex.empty() || principal.tenant_id.empty() ||
      principal.subject.empty()) {
    return InvalidArgumentError("hash and principal identity are required");
  }
  return OkStatus();
}

Status Authorize(const Principal& principal, std::string_view required_scope) {
  if (required_scope.empty() || HasScope(principal, required_scope)) {
    return OkStatus();
  }
  return absl::PermissionDeniedError("principal lacks required scope");
}

}  // namespace inferx::server::auth
