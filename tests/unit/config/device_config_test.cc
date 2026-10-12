#include "inferx/config/device_config.h"

#include <string_view>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/build_config.h"
#include "inferx/core/device.h"

namespace inferx {
namespace {

// ParseDeviceSpec ------------------------------------------------------------

TEST(ParseDeviceSpecTest, AutoYieldsAutomaticSpec) {
  const StatusOr<DeviceSpec> spec = ParseDeviceSpec("auto", {});
  ASSERT_TRUE(spec.ok()) << spec.status();
  EXPECT_TRUE(spec->automatic);
  EXPECT_TRUE(spec->devices.empty());
}

TEST(ParseDeviceSpecTest, AutoIsCaseInsensitiveAndWhitespaceTolerant) {
  for (const std::string_view name : {"AUTO", "  auto  ", "Auto"}) {
    const StatusOr<DeviceSpec> spec = ParseDeviceSpec(name, {});
    ASSERT_TRUE(spec.ok()) << name << ": " << spec.status();
    EXPECT_TRUE(spec->automatic);
  }
}

TEST(ParseDeviceSpecTest, ExplicitTypeDefaultsToOrdinalZero) {
  const StatusOr<DeviceSpec> cuda = ParseDeviceSpec("cuda", {});
  ASSERT_TRUE(cuda.ok()) << cuda.status();
  EXPECT_FALSE(cuda->automatic);
  EXPECT_EQ(cuda->devices, (std::vector<Device>{Device::Cuda(0)}));

  const StatusOr<DeviceSpec> cpu = ParseDeviceSpec("CPU", {});
  ASSERT_TRUE(cpu.ok()) << cpu.status();
  EXPECT_FALSE(cpu->automatic);
  EXPECT_EQ(cpu->devices, (std::vector<Device>{Device::Cpu()}));
}

TEST(ParseDeviceSpecTest, IdsPreserveOrderAndSetOrdinals) {
  const StatusOr<DeviceSpec> spec = ParseDeviceSpec("cuda", {2, 0, 5});
  ASSERT_TRUE(spec.ok()) << spec.status();
  EXPECT_EQ(spec->devices,
            (std::vector<Device>{Device::Cuda(2), Device::Cuda(0), Device::Cuda(5)}));
}

TEST(ParseDeviceSpecTest, CpuAcceptsOnlyEmptyOrZeroIds) {
  const StatusOr<DeviceSpec> none = ParseDeviceSpec("cpu", {});
  ASSERT_TRUE(none.ok()) << none.status();
  EXPECT_EQ(none->devices, (std::vector<Device>{Device::Cpu()}));

  const StatusOr<DeviceSpec> zero = ParseDeviceSpec("cpu", {0});
  ASSERT_TRUE(zero.ok()) << zero.status();
  EXPECT_EQ(zero->devices, (std::vector<Device>{Device::Cpu()}));
}

TEST(ParseDeviceSpecTest, RejectsMalformedFlags) {
  EXPECT_FALSE(ParseDeviceSpec("", {}).ok());        // empty type
  EXPECT_FALSE(ParseDeviceSpec("cuda:0", {}).ok());  // ordinal belongs in --device-ids
  EXPECT_FALSE(ParseDeviceSpec("tpu", {}).ok());     // unknown type
  EXPECT_FALSE(ParseDeviceSpec("auto", {0}).ok());   // ids need an explicit type
}

TEST(ParseDeviceSpecTest, RejectsInvalidIds) {
  EXPECT_FALSE(ParseDeviceSpec("cuda", {-1}).ok());    // negative
  EXPECT_FALSE(ParseDeviceSpec("cuda", {2, 2}).ok());  // duplicate
  EXPECT_FALSE(ParseDeviceSpec("cuda", {128}).ok());   // over the int8_t range
  EXPECT_FALSE(ParseDeviceSpec("cpu", {1}).ok());      // cpu has one device
  EXPECT_FALSE(ParseDeviceSpec("cpu", {0, 1}).ok());   // ditto, longer list

  // Boundary: 127 is the largest ordinal a Device can name.
  const StatusOr<DeviceSpec> edge = ParseDeviceSpec("cuda", {127});
  ASSERT_TRUE(edge.ok()) << edge.status();
  EXPECT_EQ(edge->devices, (std::vector<Device>{Device::Cuda(127)}));
}

// DeviceConfig ----------------------------------------------------------------

TEST(DeviceConfigTest, ValidateAcceptsAutoAndCpu) {
  EXPECT_TRUE(DeviceConfig{}.Validate().ok());  // default device is "auto"

  DeviceConfig cpu;
  cpu.device = "cpu";
  EXPECT_TRUE(cpu.Validate().ok());
}

TEST(DeviceConfigTest, ValidateRejectsBadFlags) {
  DeviceConfig unknown;
  unknown.device = "tpu";
  EXPECT_FALSE(unknown.Validate().ok());

  DeviceConfig ids_without_type;
  ids_without_type.device = "auto";
  ids_without_type.device_ids = {0};
  EXPECT_FALSE(ids_without_type.Validate().ok());

  DeviceConfig negative;
  negative.device = "cuda";
  negative.device_ids = {0, -1};
  EXPECT_FALSE(negative.Validate().ok());
}

// Compile-time capability: an explicit accelerator type is valid only when its
// backend was compiled in. On a CUDA build both sides are true; wherever
// CPU-only builds run, both are false.
TEST(DeviceConfigTest, ValidateRequiresBuiltBackend) {
  DeviceConfig cuda;
  cuda.device = "cuda";
  EXPECT_EQ(cuda.Validate().ok(), kHasCuda);

  DeviceConfig hip;
  hip.device = "hip";
  EXPECT_EQ(hip.Validate().ok(), kHasHip);
}

// Resolve ---------------------------------------------------------------------

TEST(DeviceConfigTest, ResolveExplicitCpu) {
  DeviceConfig cpu;
  cpu.device = "cpu";
  const StatusOr<std::vector<Device>> devices = cpu.Resolve();
  ASSERT_TRUE(devices.ok()) << devices.status();
  EXPECT_EQ(*devices, (std::vector<Device>{Device::Cpu()}));
}

// Hardware-independent: with an attached accelerator auto selects its first
// device; without one it falls back to CPU. Either way, exactly one device.
TEST(DeviceConfigTest, ResolveAutoYieldsExactlyOneDevice) {
  const StatusOr<std::vector<Device>> devices = DeviceConfig{}.Resolve();
  ASSERT_TRUE(devices.ok()) << devices.status();
  EXPECT_EQ(devices->size(), 1u);
}

}  // namespace
}  // namespace inferx
