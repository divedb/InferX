#include "inferx/server/auth/api_key_store.h"
#include "inferx/server/auth/rbac.h"

namespace inferx::server::auth {

bool HasScope(const Principal& principal, std::string_view scope) {
  return principal.scopes.find(std::string(scope)) != principal.scopes.end();
}

Status ApiKeyStore::AddHash(std::string hash_hex, Principal principal) {
  if (hash_hex.empty() || principal.tenant_id.empty() ||
      principal.subject.empty()) {
    return InvalidArgumentError("hash and principal identity are required");
  }
  std::lock_guard lock(mutex_);
  if (entries_.contains(hash_hex)) {
    return FailedPreconditionError("API-key hash already exists");
  }
  entries_.emplace(std::move(hash_hex), std::move(principal));
  return OkStatus();
}

Status ApiKeyStore::RemoveHash(std::string_view hash_hex) {
  std::lock_guard lock(mutex_);
  entries_.erase(std::string(hash_hex));
  return OkStatus();
}

StatusOr<Principal> ApiKeyStore::LookupHash(std::string_view hash_hex) const {
  std::lock_guard lock(mutex_);
  const auto it = entries_.find(std::string(hash_hex));
  if (it == entries_.end()) return absl::UnauthenticatedError("invalid API key");
  return it->second;
}

size_t ApiKeyStore::size() const {
  std::lock_guard lock(mutex_);
  return entries_.size();
}

Status Authorize(const Principal& principal, std::string_view required_scope) {
  if (required_scope.empty() || HasScope(principal, required_scope)) {
    return OkStatus();
  }
  return absl::PermissionDeniedError("principal lacks required scope");
}

}  // namespace inferx::server::auth
