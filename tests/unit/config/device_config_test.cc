#include "inferx/config/device_config.h"

#include "gtest/gtest.h"
#include "inferx/core/device.h"

namespace inferx {
namespace {

TEST(DeviceConfigTest, PrimaryDeviceResolvesTypeAndFirstId) {
  EXPECT_EQ(DeviceConfig{}.PrimaryDevice(), DeviceId::Cuda(0));

  DeviceConfig second;
  second.device_ids = {1};
  EXPECT_EQ(second.PrimaryDevice(), DeviceId::Cuda(1));

  DeviceConfig ranked;
  ranked.device_ids = {3, 1};  // Multi-id lists arrive with tensor parallelism.
  EXPECT_EQ(ranked.PrimaryDevice(), DeviceId::Cuda(3));

  DeviceConfig cpu;
  cpu.device_type = "cpu";
  EXPECT_EQ(cpu.PrimaryDevice(), DeviceId::Cpu());
}

TEST(DeviceConfigTest, ValidateRejectsStructuralErrors) {
  EXPECT_TRUE(DeviceConfig{}.Validate().ok());

  DeviceConfig unknown;
  unknown.device_type = "tpu";
  EXPECT_FALSE(unknown.Validate().ok());

  DeviceConfig negative;
  negative.device_ids = {0, -1};
  EXPECT_FALSE(negative.Validate().ok());

  DeviceConfig duplicate;
  duplicate.device_ids = {2, 2};
  EXPECT_FALSE(duplicate.Validate().ok());
}

// Compile-time capability: "cuda" must be rejected by a CPU-only build.
// On a CUDA build this asserts the pairing holds; the !kCudaBuilt branch is
// exercised wherever such builds run.
TEST(DeviceConfigTest, TypeRequiresCompiledBackend) {
  DeviceConfig cuda;
  cuda.device_type = "cuda";
  EXPECT_EQ(cuda.Validate().ok(), kCudaBuilt);

  DeviceConfig cpu;
  cpu.device_type = "cpu";
  EXPECT_TRUE(cpu.Validate().ok());
}

}  // namespace
}  // namespace inferx
