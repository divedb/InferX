#include "models/weight_upload.h"

namespace inferx::models {
namespace {

/// \brief Rounds a float32 to the nearest bfloat16 bit pattern.
///
/// Round-to-nearest-even, matching torch's float->bfloat16 conversion; a
/// bf16->fp32->bf16 round trip through the loader is exact.
uint16_t FloatToBf16Bits(float value) {
  uint32_t bits = 0;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t lsb = (bits >> 16) & 1u;
  const uint32_t rounding = 0x7fffu + lsb;
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

}  // namespace

StatusOr<Tensor> UploadBf16(const SafeTensor& tensor, DeviceId device) {
  const Shape shape(tensor.shape);
  INFERX_ASSIGN_OR_RETURN(Tensor out, Tensor::Empty(DataType::kBFloat16, shape, device));

  std::vector<uint16_t> host(static_cast<size_t>(shape.Numel()));
  for (size_t i = 0; i < host.size(); ++i) {
    host[i] = FloatToBf16Bits(tensor.data[i]);
  }
  INFERX_ASSIGN_OR_RETURN(DeviceRuntime * runtime, RuntimeFor(device));
  INFERX_RETURN_IF_ERROR(runtime->Activate());
  INFERX_RETURN_IF_ERROR(runtime->Copy(
      out.Data(), host.data(), static_cast<size_t>(out.NBytes()), CopyKind::kHostToDevice));
  return out;
}

StatusOr<Tensor> GetWeight(const SafeTensors& tensors, const std::string& name,
                           DeviceId device) {
  const SafeTensor* tensor = tensors.Find(name);
  if (tensor == nullptr) {
    return NotFoundError("checkpoint is missing tensor ", name);
  }
  return UploadBf16(*tensor, device);
}

}  // namespace inferx::models
