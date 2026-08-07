#pragma once

#include <string_view>

#include "inferx/core/status.h"
#include "inferx/server/auth/principal.h"

namespace inferx::server::auth {

Status Authorize(const Principal& principal, std::string_view required_scope);

}  // namespace inferx::server::auth
