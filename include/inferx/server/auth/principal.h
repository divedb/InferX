#pragma once

#include <string>
#include <string_view>
#include <unordered_set>

namespace inferx::server::auth {

using Scope = std::string;
using ScopeSet = std::unordered_set<Scope>;

struct Principal {
  std::string tenant_id;
  std::string subject;
  std::string key_id;
  ScopeSet scopes;
};

bool HasScope(const Principal& principal, std::string_view scope);

}  // namespace inferx::server::auth
