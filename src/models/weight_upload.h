#ifndef INFERX_MODELS_WEIGHT_UPLOAD_H_
#define INFERX_MODELS_WEIGHT_UPLOAD_H_

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/models/safetensors.h"

namespace inferx::models {

/// \brief Uploads a checkpoint tensor to `device` as bfloat16.
///
/// The SafeTensors reader stages everything as host float32; this converts
/// (losslessly for bf16 checkpoints) and copies into a fresh device tensor.
StatusOr<Tensor> UploadBf16(const SafeTensor& tensor, DeviceId device);

/// \brief Loads one named weight as bf16 on the device, or errors naming it.
StatusOr<Tensor> GetWeight(const SafeTensors& tensors, const std::string& name,
                           DeviceId device);

}  // namespace inferx::models

#endif  // INFERX_MODELS_WEIGHT_UPLOAD_H_
