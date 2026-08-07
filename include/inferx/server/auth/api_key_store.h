#pragma once

#include <atomic>
#include <memory>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/server/auth/principal.h"

namespace inferx::server::auth {

struct ApiKeyRecord {
  std::string hash_hex;
  Principal principal;
};

/// Stores only externally computed SHA-256 hex digests, never bearer tokens.
class ApiKeyStore {
 public:
  ApiKeyStore();

  Status AddHash(std::string hash_hex, Principal principal);
  Status RemoveHash(std::string_view hash_hex);
  /// Validates and atomically activates a complete key snapshot. Readers see
  /// either the old snapshot or the new one, never a partially rotated set.
  Status ReplaceAll(std::vector<ApiKeyRecord> records);
  StatusOr<Principal> LookupHash(std::string_view hash_hex) const;
  size_t size() const;

 private:
  using Entries = std::unordered_map<std::string, Principal>;

  static Status Validate(std::string_view hash_hex,
                         const Principal& principal);

  std::mutex writer_mutex_;
  std::atomic<std::shared_ptr<const Entries>> entries_;
};

}  // namespace inferx::server::auth
