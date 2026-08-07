#include "inferx/server/auth/api_key_store.h"
#include "inferx/server/auth/authenticator.h"
#include "inferx/server/auth/rbac.h"

#include <openssl/crypto.h>
#include <openssl/evp.h>

#include <array>
#include <iomanip>
#include <sstream>

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
  const Principal* matched = nullptr;
  for (const auto& [accepted_hash, principal] : *snapshot) {
    if (accepted_hash.size() != hash_hex.size()) continue;
    if (CRYPTO_memcmp(accepted_hash.data(), hash_hex.data(), hash_hex.size()) ==
        0) {
      matched = &principal;
    }
  }
  if (matched == nullptr) return absl::UnauthenticatedError("invalid API key");
  return *matched;
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

ApiKeyAuthenticator::ApiKeyAuthenticator(const ApiKeyStore* store,
                                         bool allow_anonymous)
    : store_(store), allow_anonymous_(allow_anonymous) {}

StatusOr<Principal> ApiKeyAuthenticator::Authenticate(
    std::string_view authorization) const {
  if (store_ == nullptr) return InternalError("API-key store is unavailable");
  if (allow_anonymous_ && store_->size() == 0) {
    return Principal{.tenant_id = "development",
                     .subject = "anonymous",
                     .scopes = {"inference.invoke", "models.read"}};
  }

  constexpr std::string_view prefix = "Bearer ";
  if (!authorization.starts_with(prefix) ||
      authorization.size() == prefix.size()) {
    return absl::UnauthenticatedError(
        "Authorization header must contain a Bearer token");
  }
  const std::string_view token = authorization.substr(prefix.size());
  if (token.find_first_of(" \t\r\n") != std::string_view::npos) {
    return absl::UnauthenticatedError("Bearer token is malformed");
  }

  std::array<unsigned char, EVP_MAX_MD_SIZE> digest{};
  unsigned int digest_size = 0;
  if (EVP_Digest(token.data(), token.size(), digest.data(), &digest_size,
                 EVP_sha256(), nullptr) != 1) {
    return InternalError("failed to hash API key");
  }
  std::ostringstream encoded;
  encoded << std::hex << std::setfill('0');
  for (unsigned int i = 0; i < digest_size; ++i) {
    encoded << std::setw(2) << static_cast<unsigned int>(digest[i]);
  }
  return store_->LookupHash(encoded.str());
}

Status Authorize(const Principal& principal, std::string_view required_scope) {
  if (required_scope.empty() || HasScope(principal, required_scope)) {
    return OkStatus();
  }
  return absl::PermissionDeniedError("principal lacks required scope");
}

}  // namespace inferx::server::auth
