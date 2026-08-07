#pragma once

#include <string_view>

#include "inferx/core/status.h"
#include "inferx/server/auth/api_key_store.h"

namespace inferx::server::auth {

/// Authenticates an HTTP Authorization header without retaining the bearer
/// token. Development bypass is explicit and disabled by default.
class ApiKeyAuthenticator {
 public:
  explicit ApiKeyAuthenticator(const ApiKeyStore* store,
                               bool allow_anonymous = false);

  StatusOr<Principal> Authenticate(std::string_view authorization) const;

 private:
  const ApiKeyStore* store_;
  bool allow_anonymous_;
};

}  // namespace inferx::server::auth
