#pragma once

#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>

#include "inferx/core/status.h"
#include "inferx/server/auth/principal.h"

namespace inferx::server::auth {

/// Stores only externally computed SHA-256 hex digests, never bearer tokens.
class ApiKeyStore {
 public:
  Status AddHash(std::string hash_hex, Principal principal);
  Status RemoveHash(std::string_view hash_hex);
  StatusOr<Principal> LookupHash(std::string_view hash_hex) const;
  size_t size() const;

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::string, Principal> entries_;
};

}  // namespace inferx::server::auth
