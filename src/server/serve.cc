#include "inferx/server/serve.h"

namespace inferx {

Status ServeConfig::Validate() const {
  INFERX_RETURN_IF_ERROR(server.Validate());
  INFERX_RETURN_IF_ERROR(model.device.Validate());
  INFERX_RETURN_IF_ERROR(cache.Validate());
  INFERX_RETURN_IF_ERROR(parallel.Validate());

  return OkStatus();
}

Status RunServe(const ServeConfig& config) {
  INFERX_RETURN_IF_ERROR(config.Validate());

  return UnimplementedError(
      "serve: the HTTP serving stack is not implemented yet "
      "(docs/server_design.md §14)");
}

}  // namespace inferx
