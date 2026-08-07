#pragma once

#include <memory>
#include <vector>

#include "inferx/api/openai.h"
#include "inferx/server/model_registry/registry.h"
#include "inferx/server/transport/response_writer.h"

namespace inferx::server::handlers {

struct EmbeddingsOutput {
  std::vector<std::vector<float>> values;
  uint32_t prompt_tokens = 0;
};

/// Execution-neutral embedding boundary. Implementations must run a genuine
/// embedding workload; generation logits are not an acceptable substitute.
class EmbeddingsService {
 public:
  virtual ~EmbeddingsService() = default;
  virtual folly::coro::Task<StatusOr<EmbeddingsOutput>> Embed(
      const ::inferx::api::EmbeddingsRequest& request,
      const model_registry::ModelRecord& model,
      const transport::RequestContext& context,
      folly::CancellationToken cancellation) = 0;
};

class EmbeddingsHandler final : public transport::RequestHandler {
 public:
  EmbeddingsHandler(const model_registry::Registry* registry,
                    std::shared_ptr<EmbeddingsService> service = nullptr)
      : registry_(registry), service_(std::move(service)) {}

  folly::coro::Task<void> Handle(
      transport::HttpRequest request, transport::RequestContext context,
      transport::ResponseWriter& response,
      folly::CancellationToken cancellation) override;

 private:
  const model_registry::Registry* registry_;
  std::shared_ptr<EmbeddingsService> service_;
};

}  // namespace inferx::server::handlers
