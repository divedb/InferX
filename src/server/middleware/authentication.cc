#include "inferx/server/middleware/authentication.h"

#include <boost/beast/http/field.hpp>

#include "inferx/server/auth/rbac.h"

namespace inferx::server::middleware {

BearerRouteGuard::BearerRouteGuard(
    std::shared_ptr<const auth::ApiKeyAuthenticator> authenticator)
    : authenticator_(std::move(authenticator)) {}

folly::coro::Task<Status> BearerRouteGuard::Check(
    const transport::HttpRequest& request,
    const transport::RouteMetadata& metadata,
    transport::RequestContext& context,
    folly::CancellationToken cancellation) {
  if (!metadata.authentication_required) co_return OkStatus();
  if (cancellation.isCancellationRequested()) {
    co_return absl::CancelledError("request cancelled");
  }
  if (authenticator_ == nullptr) {
    co_return InternalError("authentication middleware is not configured");
  }
  const auto header = request.find(boost::beast::http::field::authorization);
  if (header == request.end()) {
    co_return absl::UnauthenticatedError("Authorization header is required");
  }
  auto authenticated = authenticator_->Authenticate(std::string_view(
      header->value().data(), header->value().size()));
  if (!authenticated.ok()) co_return authenticated.status();
  Status authorized = auth::Authorize(*authenticated, metadata.required_scope);
  if (!authorized.ok()) co_return authorized;
  context.tenant_id = authenticated->tenant_id;
  context.subject = authenticated->subject;
  context.api_key_id = authenticated->key_id;
  context.scopes = authenticated->scopes;
  context.authenticated = true;
  co_return OkStatus();
}

}  // namespace inferx::server::middleware
