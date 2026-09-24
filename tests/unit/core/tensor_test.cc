#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/tensor.h"

namespace inferx {
namespace {

TEST(TensorToTest, SameDeviceAliasesStorage) {
  const std::vector<float> values{1.0f, 2.0f, 3.0f, 4.0f};
  auto tensor = Tensor::FromBlob(const_cast<float*>(values.data()), DataType::kFloat32,
                                 Shape({4}), DeviceId::Cpu());
  ASSERT_TRUE(tensor.ok());

  const auto same = tensor->To(DeviceId::Cpu());
  ASSERT_TRUE(same.ok());
  EXPECT_EQ(same->Data(), tensor->Data());
  EXPECT_TRUE(same->IsCpu());
}

TEST(TensorToTest, RoundTripPreservesBytes) {
  // The only other device on this box is CUDA; skip when it is absent.
  if (!RuntimeFor(DeviceId::Cuda(0)).ok()) {
    GTEST_SKIP() << "no CUDA runtime";
  }

  const std::vector<float> values{1.5f, -2.0f, 0.25f, 100.0f, -0.5f, 8.0f};
  auto host = Tensor::FromBlob(const_cast<float*>(values.data()), DataType::kFloat32,
                               Shape({2, 3}), DeviceId::Cpu());
  ASSERT_TRUE(host.ok());

  const auto device = host->To(DeviceId::Cuda(0));
  ASSERT_TRUE(device.ok()) << device.status();
  EXPECT_TRUE(device->IsCuda());
  EXPECT_EQ(device->GetDataType(), DataType::kFloat32);
  EXPECT_EQ(device->Rank(), 2);
  EXPECT_EQ(device->Numel(), 6);

  const auto back = device->To(DeviceId::Cpu());
  ASSERT_TRUE(back.ok()) << back.status();
  const float* data = back->DataAs<float>();
  ASSERT_NE(data, nullptr);
  for (std::size_t i = 0; i < values.size(); ++i) {
    EXPECT_FLOAT_EQ(data[i], values[i]) << "element " << i;
  }
}

}  // namespace
}  // namespace inferx
