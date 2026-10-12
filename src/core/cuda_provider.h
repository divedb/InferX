// Private to inferx_core: the hook the registry (device_runtime.cc) uses to
// reach the CUDA backend when INFERX_WITH_CUDA is defined. The definition
// lives in cuda/device_runtime.cc; builds without CUDA never reference it.
#pragma once

#include <memory>

#include "inferx/core/device_runtime.h"

namespace inferx {

/// \brief Returns the CUDA RuntimeProvider.
///
/// \return A provider serving DeviceType::kCuda.
std::unique_ptr<RuntimeProvider> MakeCudaProvider();

}  // namespace inferx
