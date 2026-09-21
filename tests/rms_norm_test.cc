#include "inferx/ops/rms_norm.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/ops/execution_context.h"

namespace inferx::ops {
namespace {

static_assert(sizeof(_Float16) == 2, "host half conversions need 16-bit _Float16");

// Host-side bfloat16 conversion: round-to-nearest-even into the high 16 bits
// of the float encoding. _Float16 supplies the compiler's exact fp16 rounding.
uint16_t FloatToBf16Bits(float f) {
  uint32_t bits = 0;
  std::memcpy(&bits, &f, sizeof(bits));
  const uint32_t rounding = 0x7fff + ((bits >> 16) & 1);
  return static_cast<uint16_t>((bits + rounding) >> 16);
}

float Bf16BitsToFloat(uint16_t h) {
  const uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f = 0.0f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

uint16_t FloatToF16Bits(float f) {
  const _Float16 h = static_cast<_Float16>(f);
  uint16_t bits = 0;
  std::memcpy(&bits, &h, sizeof(bits));
  return bits;
}

float F16BitsToFloat(uint16_t h) {
  _Float16 v = 0;
  std::memcpy(&v, &h, sizeof(v));
  return static_cast<float>(v);
}

// Per-dtype encoding and comparison tolerance for the reference checks. The
// reference is computed in float from the encode/decode round trip, so only
// the output rounding of the tested dtype sets the tolerance.
struct Bf16Policy {
  using Bits = uint16_t;
  static constexpr DataType kDtype = DataType::kBFloat16;
  static constexpr float kRel = 1e-2f;
  static constexpr float kAbs = 1e-2f;
  static Bits Encode(float v) { return FloatToBf16Bits(v); }
  static float Decode(Bits h) { return Bf16BitsToFloat(h); }
};

struct F16Policy {
  using Bits = uint16_t;
  static constexpr DataType kDtype = DataType::kFloat16;
  static constexpr float kRel = 2e-3f;
  static constexpr float kAbs = 2e-3f;
  static Bits Encode(float v) { return FloatToF16Bits(v); }
  static float Decode(Bits h) { return F16BitsToFloat(h); }
};

struct F32Policy {
  using Bits = float;
  static constexpr DataType kDtype = DataType::kFloat;
  static constexpr float kRel = 1e-4f;
  static constexpr float kAbs = 1e-4f;
  static Bits Encode(float v) { return v; }
  static float Decode(Bits v) { return v; }
};

// One execution lane (runtime + stream) on a device, shared by the per-device
// fixtures below.
class RmsNormTest : public ::testing::Test {
 protected:
  void SetUpOn(DeviceId device) {
    auto runtime = RuntimeFor(device);
    ASSERT_TRUE(runtime.ok());
    runtime_ = *runtime;
    ASSERT_TRUE(runtime_->Activate().ok());
    auto stream = runtime_->CreateStream();
    ASSERT_TRUE(stream.ok());
    stream_ = *stream;
    device_ = device;
  }

  void TearDown() override {
    if (runtime_ != nullptr && stream_.handle != nullptr) {
      (void)runtime_->SynchronizeStream(stream_);
      (void)runtime_->DestroyStream(stream_);
    }
  }

  /// \brief Runs RmsNorm out of place on deterministic random data, checks
  ///        every element against a float reference computed from the same
  ///        dtype-rounded inputs, and confirms `x` was left untouched.
  template <typename Policy>
  void ExpectMatchesReference(int rows, int dim, float eps, bool plus_one) {
    std::mt19937 rng(/*seed=*/1234 + rows * 31 + dim);
    std::uniform_real_distribution<float> values(-2.0f, 2.0f);
    std::uniform_real_distribution<float> scales(0.25f, 1.75f);

    std::vector<typename Policy::Bits> x_bits(static_cast<size_t>(rows) * dim);
    std::vector<typename Policy::Bits> w_bits(dim);
    std::vector<float> x_ref(x_bits.size()), w_ref(dim);
    for (size_t i = 0; i < x_bits.size(); ++i) {
      x_bits[i] = Policy::Encode(values(rng));
      x_ref[i] = Policy::Decode(x_bits[i]);
    }
    for (int j = 0; j < dim; ++j) {
      w_bits[j] = Policy::Encode(scales(rng));
      w_ref[j] = Policy::Decode(w_bits[j]);
    }

    auto x_alloc = Tensor::Empty(Policy::kDtype, Shape({rows, dim}), device_);
    auto w_alloc = Tensor::Empty(Policy::kDtype, Shape({dim}), device_);
    auto out_alloc = Tensor::Empty(Policy::kDtype, Shape({rows, dim}), device_);
    ASSERT_TRUE(x_alloc.ok());
    ASSERT_TRUE(w_alloc.ok());
    ASSERT_TRUE(out_alloc.ok());
    Tensor x = *x_alloc;
    const Tensor w = *w_alloc;
    Tensor out = *out_alloc;
    const size_t bytes = x_bits.size() * sizeof(typename Policy::Bits);
    ASSERT_TRUE(runtime_->Copy(x.Data(), x_bits.data(), bytes, CopyKind::kHostToDevice).ok());
    ASSERT_TRUE(runtime_
                    ->Copy(w.Data(), w_bits.data(),
                           w_bits.size() * sizeof(typename Policy::Bits),
                           CopyKind::kHostToDevice)
                    .ok());

    RMSNormConfig config;
    config.eps = eps;
    config.plus_one_weight = plus_one;
    ExecutionContext exec_ctx(*runtime_, stream_);
    EXPECT_TRUE(RmsNorm(exec_ctx, x, w, out, config).ok());
    ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());

    std::vector<typename Policy::Bits> got(x_bits.size()), x_after(x_bits.size());
    ASSERT_TRUE(runtime_->Copy(got.data(), out.Data(), bytes, CopyKind::kDeviceToHost).ok());
    ASSERT_TRUE(runtime_->Copy(x_after.data(), x.Data(), bytes, CopyKind::kDeviceToHost).ok());
    EXPECT_EQ(x_after, x_bits);  // Out of place: the input stays intact.
    for (int i = 0; i < rows; ++i) {
      double mean_square = 0.0;
      for (int j = 0; j < dim; ++j) {
        const float v = x_ref[static_cast<size_t>(i) * dim + j];
        mean_square += static_cast<double>(v) * v;
      }
      mean_square /= dim;
      const float inv_rms = 1.0f / std::sqrt(static_cast<float>(mean_square) + eps);
      for (int j = 0; j < dim; ++j) {
        const size_t idx = static_cast<size_t>(i) * dim + j;
        const float scale = plus_one ? 1.0f + w_ref[j] : w_ref[j];
        const float expected = x_ref[idx] * inv_rms * scale;
        const float actual = Policy::Decode(got[idx]);
        // One rounding of the output plus accumulated float error.
        EXPECT_NEAR(actual, expected, Policy::kAbs + Policy::kRel * std::fabs(expected))
            << "row " << i << " column " << j;
      }
    }
  }

  ExecutionContext ctx() { return ExecutionContext(*runtime_, stream_); }

  DeviceRuntime* runtime_ = nullptr;
  Stream stream_;
  DeviceId device_ = DeviceId::Cpu();
};

class CudaRmsNormTest : public RmsNormTest {
 protected:
  void SetUp() override { SetUpOn(DeviceId::Cuda(0)); }
};

class CpuRmsNormTest : public RmsNormTest {
 protected:
  void SetUp() override { SetUpOn(DeviceId::Cpu()); }
};

TEST_F(CudaRmsNormTest, StandardMatchesReference) {
  ExpectMatchesReference<Bf16Policy>(/*rows=*/7, /*dim=*/96, /*eps=*/1e-6f, /*plus_one=*/false);
}

TEST_F(CudaRmsNormTest, LargeHiddenSizeMatchesReference) {
  ExpectMatchesReference<Bf16Policy>(/*rows=*/3, /*dim=*/4096, /*eps=*/1e-6f,
                                     /*plus_one=*/false);
}

TEST_F(CudaRmsNormTest, OddHiddenSizeMatchesReference) {
  // dim not divisible by the kernel's vector width exercises the scalar path.
  ExpectMatchesReference<Bf16Policy>(/*rows=*/5, /*dim=*/100, /*eps=*/1e-3f,
                                     /*plus_one=*/false);
}

TEST_F(CudaRmsNormTest, PlusOneWeightMatchesReference) {
  ExpectMatchesReference<Bf16Policy>(/*rows=*/4, /*dim=*/256, /*eps=*/1e-6f, /*plus_one=*/true);
}

TEST_F(CudaRmsNormTest, Float16MatchesReference) {
  ExpectMatchesReference<F16Policy>(/*rows=*/6, /*dim=*/512, /*eps=*/1e-6f, /*plus_one=*/false);
}

TEST_F(CudaRmsNormTest, Float16PlusOneWeightMatchesReference) {
  ExpectMatchesReference<F16Policy>(/*rows=*/4, /*dim=*/256, /*eps=*/1e-6f, /*plus_one=*/true);
}

TEST_F(CudaRmsNormTest, Float32MatchesReference) {
  ExpectMatchesReference<F32Policy>(/*rows=*/6, /*dim=*/512, /*eps=*/1e-6f, /*plus_one=*/false);
  ExpectMatchesReference<F32Policy>(/*rows=*/3, /*dim=*/4096, /*eps=*/1e-6f,
                                    /*plus_one=*/false);
}

TEST_F(CudaRmsNormTest, Float32PlusOneWeightMatchesReference) {
  ExpectMatchesReference<F32Policy>(/*rows=*/4, /*dim=*/256, /*eps=*/1e-6f, /*plus_one=*/true);
}

TEST_F(CudaRmsNormTest, RejectsWeightLengthMismatch) {
  auto x = Tensor::Empty(DataType::kBFloat16, Shape({4, 8}), device_);
  auto w = Tensor::Empty(DataType::kBFloat16, Shape({7}), device_);
  auto out = Tensor::Empty(DataType::kBFloat16, Shape({4, 8}), device_);
  ASSERT_TRUE(x.ok() && w.ok() && out.ok());
  ExecutionContext exec_ctx(*runtime_, stream_);
  const Status status = RmsNorm(exec_ctx, *x, *w, *out, RMSNormConfig{});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(CudaRmsNormTest, RejectsDtypeMismatch) {
  auto x = Tensor::Empty(DataType::kBFloat16, Shape({4, 8}), device_);
  auto w = Tensor::Empty(DataType::kFloat16, Shape({8}), device_);
  auto out = Tensor::Empty(DataType::kBFloat16, Shape({4, 8}), device_);
  ASSERT_TRUE(x.ok() && w.ok() && out.ok());
  ExecutionContext exec_ctx(*runtime_, stream_);
  const Status status = RmsNorm(exec_ctx, *x, *w, *out, RMSNormConfig{});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(CudaRmsNormTest, RejectsOutputShapeMismatch) {
  auto x = Tensor::Empty(DataType::kBFloat16, Shape({4, 8}), device_);
  auto w = Tensor::Empty(DataType::kBFloat16, Shape({8}), device_);
  auto out = Tensor::Empty(DataType::kBFloat16, Shape({4, 9}), device_);
  ASSERT_TRUE(x.ok() && w.ok() && out.ok());
  ExecutionContext exec_ctx(*runtime_, stream_);
  const Status status = RmsNorm(exec_ctx, *x, *w, *out, RMSNormConfig{});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(CudaRmsNormTest, RejectsUnsupportedDtype) {
  auto x = Tensor::Empty(DataType::kDouble, Shape({4, 8}), device_);
  auto w = Tensor::Empty(DataType::kDouble, Shape({8}), device_);
  auto out = Tensor::Empty(DataType::kDouble, Shape({4, 8}), device_);
  ASSERT_TRUE(x.ok() && w.ok() && out.ok());
  ExecutionContext exec_ctx(*runtime_, stream_);
  const Status status = RmsNorm(exec_ctx, *x, *w, *out, RMSNormConfig{});
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
}

TEST_F(CudaRmsNormTest, RejectsForeignDevice) {
  auto x = Tensor::Empty(DataType::kBFloat16, Shape({4, 8}), DeviceId::Cpu());
  auto w = Tensor::Empty(DataType::kBFloat16, Shape({8}), DeviceId::Cpu());
  auto out = Tensor::Empty(DataType::kBFloat16, Shape({4, 8}), DeviceId::Cpu());
  ASSERT_TRUE(x.ok() && w.ok() && out.ok());
  ExecutionContext exec_ctx(*runtime_, stream_);
  const Status status = RmsNorm(exec_ctx, *x, *w, *out, RMSNormConfig{});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(CpuRmsNormTest, StandardMatchesReference) {
  ExpectMatchesReference<F32Policy>(/*rows=*/7, /*dim=*/96, /*eps=*/1e-6f, /*plus_one=*/false);
  ExpectMatchesReference<Bf16Policy>(/*rows=*/5, /*dim=*/100, /*eps=*/1e-3f,
                                     /*plus_one=*/false);
  ExpectMatchesReference<F16Policy>(/*rows=*/6, /*dim=*/512, /*eps=*/1e-6f, /*plus_one=*/false);
}

TEST_F(CpuRmsNormTest, LargeHiddenSizeMatchesReference) {
  ExpectMatchesReference<F32Policy>(/*rows=*/3, /*dim=*/4096, /*eps=*/1e-6f,
                                    /*plus_one=*/false);
  ExpectMatchesReference<Bf16Policy>(/*rows=*/3, /*dim=*/4096, /*eps=*/1e-6f,
                                     /*plus_one=*/false);
}

TEST_F(CpuRmsNormTest, PlusOneWeightMatchesReference) {
  ExpectMatchesReference<F32Policy>(/*rows=*/4, /*dim=*/256, /*eps=*/1e-6f, /*plus_one=*/true);
  ExpectMatchesReference<Bf16Policy>(/*rows=*/4, /*dim=*/256, /*eps=*/1e-6f, /*plus_one=*/true);
  ExpectMatchesReference<F16Policy>(/*rows=*/4, /*dim=*/256, /*eps=*/1e-6f, /*plus_one=*/true);
}

TEST_F(CpuRmsNormTest, RejectsWeightLengthMismatch) {
  auto x = Tensor::Empty(DataType::kBFloat16, Shape({4, 8}), device_);
  auto w = Tensor::Empty(DataType::kBFloat16, Shape({7}), device_);
  auto out = Tensor::Empty(DataType::kBFloat16, Shape({4, 8}), device_);
  ASSERT_TRUE(x.ok() && w.ok() && out.ok());
  ExecutionContext exec_ctx(*runtime_, stream_);
  const Status status = RmsNorm(exec_ctx, *x, *w, *out, RMSNormConfig{});
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
}

TEST_F(CpuRmsNormTest, RejectsUnsupportedDtype) {
  auto x = Tensor::Empty(DataType::kDouble, Shape({4, 8}), device_);
  auto w = Tensor::Empty(DataType::kDouble, Shape({8}), device_);
  auto out = Tensor::Empty(DataType::kDouble, Shape({4, 8}), device_);
  ASSERT_TRUE(x.ok() && w.ok() && out.ok());
  ExecutionContext exec_ctx(*runtime_, stream_);
  const Status status = RmsNorm(exec_ctx, *x, *w, *out, RMSNormConfig{});
  EXPECT_EQ(status.code(), absl::StatusCode::kUnimplemented);
}

}  // namespace
}  // namespace inferx::ops
