#include "inferx/server/handlers/embeddings_handler.h"

#include <boost/beast/http/status.hpp>

#include "inferx/server/api/error_mapping.h"
#include "inferx/support/json.h"

namespace inferx::server::handlers {
namespace {

folly::coro::Task<void> Write(
    const transport::HttpRequest& request, transport::ResponseWriter& writer,
    unsigned status, std::string body, folly::CancellationToken cancellation) {
  transport::HttpResponse response{
      static_cast<boost::beast::http::status>(status), request.version()};
  response.keep_alive(request.keep_alive());
  response.set(boost::beast::http::field::content_type, "application/json");
  response.body() = std::move(body);
  response.prepare_payload();
  (void)co_await writer.WriteResponse(std::move(response), cancellation);
}

folly::coro::Task<void> WriteError(
    const transport::HttpRequest& request, transport::ResponseWriter& writer,
    const Status& status, std::string_view param,
    folly::CancellationToken cancellation) {
  auto error = api::MapStatus(status, param);
  co_await Write(request, writer, error.status, error.Json({}), cancellation);
}

}  // namespace

folly::coro::Task<void> EmbeddingsHandler::Handle(
    transport::HttpRequest request, transport::RequestContext context,
    transport::ResponseWriter& response,
    folly::CancellationToken cancellation) {
  if (!context.authenticated) {
    co_await WriteError(request, response,
                        absl::UnauthenticatedError(
                            "authenticated request context is required"),
                        {}, cancellation);
    co_return;
  }
  auto parsed = ::inferx::api::ParseEmbeddingsRequest(request.body());
  if (!parsed.ok()) {
    co_await WriteError(request, response, parsed.status(), {}, cancellation);
    co_return;
  }
  if (registry_ == nullptr) {
    co_await WriteError(request, response,
                        InternalError("model registry is not configured"), {},
                        cancellation);
    co_return;
  }
  auto model = registry_->Resolve(parsed->model, context.tenant_id);
  if (!model.ok()) {
    co_await WriteError(request, response, model.status(), "model",
                        cancellation);
    co_return;
  }
  if (!model->supports_embeddings) {
    co_await WriteError(request, response,
                        UnimplementedError(
                            "model does not support embedding workloads"),
                        "model", cancellation);
    co_return;
  }
  if (parsed->input.size() > model->embedding_max_batch_size) {
    co_await WriteError(request, response,
                        UnimplementedError("embedding batch size is unsupported"),
                        "input", cancellation);
    co_return;
  }
  if (!model->embedding_encoding_formats.contains(parsed->encoding_format)) {
    co_await WriteError(request, response,
                        UnimplementedError("embedding encoding is unsupported"),
                        "encoding_format", cancellation);
    co_return;
  }
  if (parsed->encoding_format != "float") {
    co_await WriteError(request, response,
                        UnimplementedError(
                            "embedding encoding is not implemented"),
                        "encoding_format", cancellation);
    co_return;
  }
  const uint32_t dimensions = parsed->has_dimensions
                                  ? static_cast<uint32_t>(parsed->dimensions)
                                  : model->embedding_dimensions;
  if (dimensions == 0 || dimensions != model->embedding_dimensions) {
    co_await WriteError(request, response,
                        UnimplementedError(
                            "embedding dimensions are unsupported"),
                        "dimensions", cancellation);
    co_return;
  }
  if (service_ == nullptr) {
    co_await WriteError(
        request, response,
        UnimplementedError("embedding execution backend is unavailable"),
        "model", cancellation);
    co_return;
  }

  auto output = co_await service_->Embed(*parsed, *model, context, cancellation);
  if (!output.ok()) {
    co_await WriteError(request, response, output.status(), {}, cancellation);
    co_return;
  }
  if (output->values.size() != parsed->input.size()) {
    co_await WriteError(request, response,
                        InternalError("embedding backend changed batch size"),
                        {}, cancellation);
    co_return;
  }
  std::string body = "{\"object\":\"list\",\"model\":";
  AppendJsonString(parsed->model, &body);
  body += ",\"data\":[";
  for (size_t index = 0; index < output->values.size(); ++index) {
    if (output->values[index].size() != dimensions) {
      co_await WriteError(
          request, response,
          InternalError("embedding backend returned invalid dimensions"), {},
          cancellation);
      co_return;
    }
    if (index != 0) body += ',';
    body += "{\"object\":\"embedding\",\"index\":" +
            std::to_string(index) + ",\"embedding\":[";
    for (size_t value = 0; value < output->values[index].size(); ++value) {
      if (value != 0) body += ',';
      body += std::to_string(output->values[index][value]);
    }
    body += "]}";
  }
  body += "],\"usage\":{\"prompt_tokens\":" +
          std::to_string(output->prompt_tokens) + ",\"total_tokens\":" +
          std::to_string(output->prompt_tokens) + "}}";
  co_await Write(request, response, 200, std::move(body), cancellation);
}

}  // namespace inferx::server::handlers
