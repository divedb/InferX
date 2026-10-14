// GreedyArgmax tie semantics at vocabulary scale. vLLM's greedy path is
// torch.argmax, which returns the FIRST maximal index on ties; the engine
// must match so exact BF16 ties resolve to the same token on both stacks.
// The cases use the exact ties observed in the vLLM parity investigation
// (tokens 334/56177 at 23.5) and the golden trace (9625/15344 at 16.625).
#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/op_context.h"
#include "inferx/ops/sampling.h"

namespace inferx {
namespace ops {
namespace {

class GreedyArgmaxTieTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto runtime = RuntimeFor(Device::Cuda(0));
    ASSERT_TRUE(runtime.ok()) << runtime.status();
    runtime_ = *runtime;
    ASSERT_TRUE(runtime_->Activate().ok());
    stream_ = *runtime_->Streams().CreateStream(StreamPriority::kNormal);
  }

  /// Runs GreedyArgmax over one bf16 row where `tied` pairs all share
  /// `value`; returns the sampled index.
  int32_t ArgmaxOf(const std::vector<uint16_t>& row_bits,
                   const std::vector<std::pair<int64_t, uint16_t>>& forced) {
    std::vector<uint16_t> bits = row_bits;
    for (const auto& [index, value] : forced) bits[index] = value;
    const int64_t vocab = static_cast<int64_t>(bits.size());
    auto logits = Tensor::Empty(DataType::kBFloat16, Shape({1, vocab}),
                                Device::Cuda(0));
    EXPECT_TRUE(logits.ok()) << logits.status();
    auto values = Tensor::Empty(DataType::kFloat32, Shape({1, 40}), Device::Cuda(0));
    auto indices = Tensor::Empty(DataType::kInt32, Shape({1, 40}), Device::Cuda(0));
    auto out = Tensor::Empty(DataType::kInt32, Shape({1}), Device::Cuda(0));
    EXPECT_TRUE(runtime_->Memory()
                    .Copy(logits->Data(), bits.data(), bits.size() * 2, CopyKind::kHostToDevice)
                    .ok());
    OpContext ctx(*runtime_, stream_.Get());
    EXPECT_TRUE(GreedyArgmax(ctx, *logits, *values, *indices, *out).ok());
    // The compute stream is non-blocking, so the host read must wait for it
    // explicitly before the synchronous default-stream copy below.
    EXPECT_TRUE(runtime_->Streams().SynchronizeStream(stream_.Get()).ok());
    int32_t result = -1;
    EXPECT_TRUE(runtime_->Memory()
                    .Copy(&result, out->Data(), sizeof(result), CopyKind::kDeviceToHost)
                    .ok());
    return result;
  }

  DeviceRuntime* runtime_ = nullptr;
  OwnedStream stream_{};
};

TEST_F(GreedyArgmaxTieTest, ExactTiesResolveToLowestIndex) {
  constexpr int64_t kVocab = 151936;
  // bf16(23.5) = 0x41BC, bf16(16.625) = 0x4185.
  const uint16_t base = 0x3F80;  // 1.0 everywhere else.

  const std::vector<uint16_t> row(kVocab, base);
  // The parity-probe tie: 334 and 56177 both at 23.5; lowest wins.
  EXPECT_EQ(ArgmaxOf(row, {{334, 0x41BC}, {56177, 0x41BC}}), 334);
  // Order of appearance must not matter.
  EXPECT_EQ(ArgmaxOf(row, {{56177, 0x41BC}, {334, 0x41BC}}), 334);
  // The golden-trace tie: 9625 and 15344 both at 16.625; lowest wins.
  EXPECT_EQ(ArgmaxOf(row, {{9625, 0x4185}, {15344, 0x4185}}), 9625);
  // A three-way tie spanning kernel partials (vocab/4096 = 38 parts).
  EXPECT_EQ(ArgmaxOf(row, {{100, 0x41BC}, {50000, 0x41BC}, {150000, 0x41BC}}), 100);
  // Strictly larger single candidate beats the tied pair.
  EXPECT_EQ(ArgmaxOf(row, {{334, 0x41BC}, {56177, 0x41BC}, {999, 0x4200}}), 999);
}

}  // namespace
}  // namespace ops
}  // namespace inferx
