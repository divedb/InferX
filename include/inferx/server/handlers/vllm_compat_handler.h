#pragma once

#include <string>
#include <utility>

#include "inferx/server/model_registry/registry.h"
#include "inferx/server/tokenization/tokenization_service.h"
#include "inferx/server/transport/response_writer.h"

/// The endpoints a vLLM client expects beyond the OpenAI surface: /version,
/// /ping, root-level /tokenize and /detokenize, and honest 501s for the
/// pooling-model APIs InferX does not serve.
namespace inferx::server::handlers {

/// `GET /version`, vLLM's shape: `{"version":"..."}`.
class VersionHandler final : public transport::RequestHandler {
 public:
  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;
};

/// `GET`/`POST /ping`: an unconditional empty-object 200, vLLM parity.
class PingHandler final : public transport::RequestHandler {
 public:
  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;
};

/// Answers 501 with an OpenAI-shaped error naming the request path and the
/// feature it would have needed. Registered on paths a vLLM client may probe
/// so it learns "never here" rather than the 404 that means "wrong URL".
class NotSupportedHandler final : public transport::RequestHandler {
 public:
  explicit NotSupportedHandler(std::string feature)
      : feature_(std::move(feature)) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  std::string feature_;
};

/// vLLM-style `POST /tokenize`. Kept separate from the /v1/tokenize handler
/// because the request and response contracts differ, not just field names.
class VllmTokenizeHandler final : public transport::RequestHandler {
 public:
  VllmTokenizeHandler(const model_registry::Registry* registry,
                      tokenization::TokenizationService* tokenization)
      : registry_(registry), tokenization_(tokenization) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const model_registry::Registry* registry_;
  tokenization::TokenizationService* tokenization_;
};

/// vLLM-style `POST /detokenize`: `{"model","tokens":[...]}` to
/// `{"prompt":"..."}` via a whole-sequence decode.
class DetokenizeHandler final : public transport::RequestHandler {
 public:
  DetokenizeHandler(const model_registry::Registry* registry,
                    tokenization::TokenizationService* tokenization)
      : registry_(registry), tokenization_(tokenization) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const model_registry::Registry* registry_;
  tokenization::TokenizationService* tokenization_;
};

}  // namespace inferx::server::handlers
