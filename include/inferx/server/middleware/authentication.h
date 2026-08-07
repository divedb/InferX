#pragma once

#include <memory>

#include "inferx/server/auth/authenticator.h"
#include "inferx/server/transport/routes.h"

namespace inferx::server::middleware {

class BearerRouteGuard final : public transport::RouteGuard {
 public:
  explicit BearerRouteGuard(
      std::shared_ptr<const auth::ApiKeyAuthenticator> authenticator);

  folly::coro::Task<Status> Check(
      const transport::HttpRequest& request,
      const transport::RouteMetadata& metadata,
      folly::CancellationToken cancellation) override;

 private:
  std::shared_ptr<const auth::ApiKeyAuthenticator> authenticator_;
};

}  // namespace inferx::server::middleware
