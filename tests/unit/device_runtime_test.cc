#include "inferx/core/device_runtime.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <type_traits>

#include "inferx/comm/communicator.h"
#include "inferx/kernels/flashinfer_attention.h"
#include "inferx/kernels/flashinfer_prefill.h"
#include "inferx/kernels/gemm.h"
#include "inferx/kernels/gpt_oss.h"
#include "inferx/kernels/layers.h"
#include "inferx/kernels/mla.h"
#include "inferx/kernels/moe.h"
#include "inferx/kernels/mxfp4.h"
#include "inferx/kernels/mxfp4_gemm.h"
#include "inferx/kernels/quantize.h"
#include "inferx/kernels/w4a16_gemm.h"

namespace inferx {
namespace {

static_assert(std::is_trivially_copyable_v<Stream>);
static_assert(sizeof(Stream) == sizeof(void*));

TEST(DeviceIdTest, NamesAllSupportedKinds) {
  EXPECT_EQ(DeviceId::Cpu().ToString(), "cpu");
  EXPECT_EQ(DeviceId::Cuda(1).ToString(), "cuda:1");
  EXPECT_EQ(DeviceId::Rocm(2).ToString(), "rocm:2");
  EXPECT_EQ(DeviceId::Ascend(3).ToString(), "ascend:3");
  EXPECT_FALSE(DeviceId::Cpu().IsAccelerator());
  EXPECT_TRUE(DeviceId::Rocm(0).IsAccelerator());
}

TEST(DeviceRuntimeTest, CpuImplementsTheRuntimeContract) {
  auto runtime_or = RuntimeFor(DeviceId::Cpu());
  ASSERT_TRUE(runtime_or.ok()) << runtime_or.status();
  DeviceRuntime* runtime = *runtime_or;
  EXPECT_EQ(runtime->kind(), DeviceKind::kCpu);
  EXPECT_EQ(runtime->DeviceCount(), 1);

  auto allocation_or = runtime->Allocate(DeviceId::Cpu(), 4);
  ASSERT_TRUE(allocation_or.ok()) << allocation_or.status();
  void* allocation = *allocation_or;
  std::array<std::byte, 4> source{std::byte{1}, std::byte{2}, std::byte{3},
                                  std::byte{4}};
  ASSERT_TRUE(runtime
                  ->Copy(allocation, source.data(), source.size(),
                         CopyKind::kDeviceToDevice)
                  .ok());
  EXPECT_EQ(*static_cast<std::byte*>(allocation), std::byte{1});

  auto stream_or = runtime->CreateStream(DeviceId::Cpu());
  ASSERT_TRUE(stream_or.ok()) << stream_or.status();
  Stream stream = *stream_or;
  auto event_or = runtime->CreateEvent(false);
  ASSERT_TRUE(event_or.ok()) << event_or.status();
  DeviceEvent event = *event_or;
  EXPECT_TRUE(runtime->RecordEvent(event, stream).ok());
  auto ready = runtime->QueryEvent(event);
  ASSERT_TRUE(ready.ok()) << ready.status();
  EXPECT_TRUE(*ready);
  EXPECT_TRUE(runtime->Free(DeviceId::Cpu(), allocation).ok());
}

#ifndef INFERX_WITH_CUDA
TEST(DeviceRuntimeTest, MissingBackendFailsClearly) {
  auto runtime = RuntimeFor(DeviceId::Cuda(0));
  EXPECT_FALSE(runtime.ok());
  EXPECT_NE(runtime.status().message().find("cuda:0"), std::string::npos);
}
#endif

}  // namespace
}  // namespace inferx
