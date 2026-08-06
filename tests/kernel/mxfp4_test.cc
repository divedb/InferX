// MXFP4 decode, against a golden file produced by HuggingFace's own decoder.
//
// Phase 1 of docs/gpt-oss-20b-llm.md. The format has one detail with no answer
// from first principles -- whether the low or the high nibble of a byte holds
// the first value -- and getting it backwards produces weights that are wrong
// and still look like a language model. So it is settled against a reference
// rather than reasoned about, and this test is where that settlement lives.
//
// The comparison is **exact**, which is unusual here and is a property of the
// format rather than an act of confidence. Every FP4 value needs at most one
// explicit mantissa bit, bf16 has seven, and the block scale is a power of two,
// which moves the exponent without touching the mantissa. So there is no
// rounding anywhere in the decode and nothing for a tolerance to absorb: a
// single differing bit means a wrong nibble, a wrong scale or a wrong index.
//
// Regenerate the golden file with:
//   scripts/gen_mxfp4_golden.py <gpt-oss-ckpt> testdata/gptoss_mxfp4_golden.bin
//
// It is derived data and is not committed; the test skips without it.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor_view.h"
#include "inferx/kernels/mxfp4.h"
#include "inferx/kernels/mxfp4_gemm.h"
#include "inferx/model/safetensors.h"

namespace inferx {
namespace {

using bf16 = __nv_bfloat16;

constexpr int kBytesPerBlock = 16;
constexpr int kValuesPerBlock = 32;

// One decoded weight slice: what came out of the checkpoint, and what
// transformers says it means.
struct GoldenCase {
  std::string label;
  int64_t rows = 0;
  int64_t num_blocks = 0;
  std::vector<uint8_t> blocks;
  std::vector<uint8_t> scales;
  std::vector<float> expected;
};

std::string GoldenPath() {
  if (const char* env = std::getenv("INFERX_TEST_MXFP4_GOLDEN")) return env;
  return "testdata/gptoss_mxfp4_golden.bin";
}

// Reads the 'IXM4' container written by scripts/gen_mxfp4_golden.py. Returns
// false when the file is simply absent, which is the ordinary case on a machine
// that has never had the checkpoint.
bool LoadGolden(std::vector<GoldenCase>* out, std::string* why) {
  const std::string path = GoldenPath();

  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    *why = "no golden file at " + path;
    return false;
  }

  auto fail = [&](const char* message) {
    *why = path + ": " + message;
    std::fclose(f);
    return false;
  };

  char magic[4];
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "IXM4", 4) != 0) {
    return fail("bad magic");
  }

  uint32_t version = 0;
  uint32_t cases = 0;
  if (std::fread(&version, 4, 1, f) != 1 || std::fread(&cases, 4, 1, f) != 1) {
    return fail("truncated header");
  }
  if (version != 1) return fail("unsupported version");

  for (uint32_t c = 0; c < cases; ++c) {
    GoldenCase gc;

    uint32_t label_len = 0;
    if (std::fread(&label_len, 4, 1, f) != 1) return fail("truncated label");
    gc.label.resize(label_len);
    if (label_len > 0 && std::fread(gc.label.data(), 1, label_len, f) != label_len) {
      return fail("truncated label body");
    }

    uint32_t rows = 0;
    uint32_t num_blocks = 0;
    if (std::fread(&rows, 4, 1, f) != 1 || std::fread(&num_blocks, 4, 1, f) != 1) {
      return fail("truncated shape");
    }

    gc.rows = rows;
    gc.num_blocks = num_blocks;

    const size_t block_bytes =
        static_cast<size_t>(rows) * num_blocks * kBytesPerBlock;
    const size_t scale_bytes = static_cast<size_t>(rows) * num_blocks;
    const size_t values = static_cast<size_t>(rows) * num_blocks * kValuesPerBlock;

    gc.blocks.resize(block_bytes);
    gc.scales.resize(scale_bytes);
    gc.expected.resize(values);

    if (std::fread(gc.blocks.data(), 1, block_bytes, f) != block_bytes ||
        std::fread(gc.scales.data(), 1, scale_bytes, f) != scale_bytes ||
        std::fread(gc.expected.data(), 4, values, f) != values) {
      return fail("truncated case body");
    }

    out->push_back(std::move(gc));
  }

  std::fclose(f);
  return true;
}

class Mxfp4Test : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

    std::string why;
    if (!LoadGolden(&cases_, &why)) {
      GTEST_SKIP() << why << "; regenerate with scripts/gen_mxfp4_golden.py";
    }
    ASSERT_FALSE(cases_.empty());
  }

  std::vector<GoldenCase> cases_;
};

// Uploads a case and decodes it, returning the device output as bf16.
std::vector<bf16> DecodeOnDevice(const GoldenCase& gc, bool gate_up) {
  auto blocks_buf = DeviceBuffer::Allocate(gc.blocks.size(), DeviceId::Cuda(0));
  auto scales_buf = DeviceBuffer::Allocate(gc.scales.size(), DeviceId::Cuda(0));
  auto out_buf = DeviceBuffer::Allocate(gc.expected.size() * sizeof(bf16),
                                        DeviceId::Cuda(0));
  EXPECT_TRUE(blocks_buf.ok() && scales_buf.ok() && out_buf.ok());

  EXPECT_EQ(cudaMemcpy(blocks_buf->data(), gc.blocks.data(), gc.blocks.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  EXPECT_EQ(cudaMemcpy(scales_buf->data(), gc.scales.data(), gc.scales.size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  // Poison the output, so a kernel that writes nothing fails rather than
  // matching a zeroed buffer against a zeroed expectation.
  EXPECT_EQ(cudaMemset(out_buf->data(), 0x7F, out_buf->size()), cudaSuccess);

  auto blocks = TensorView::Create(
      blocks_buf->data(), DataType::kUInt8,
      Shape({gc.rows, gc.num_blocks, kBytesPerBlock}), DeviceId::Cuda(0));
  auto scales = TensorView::Create(scales_buf->data(), DataType::kUInt8,
                                   Shape({gc.rows, gc.num_blocks}),
                                   DeviceId::Cuda(0));
  auto out = TensorView::Create(
      out_buf->data(), DataType::kBFloat16,
      Shape({gc.rows, gc.num_blocks * kValuesPerBlock}), DeviceId::Cuda(0));
  EXPECT_TRUE(blocks.ok() && scales.ok() && out.ok());

  const Status s =
      gate_up ? kernels::DequantizeMxfp4GateUpToBf16(*blocks, *scales, *out)
              : kernels::DequantizeMxfp4ToBf16(*blocks, *scales, *out);
  EXPECT_TRUE(s.ok()) << s;
  EXPECT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<bf16> got(gc.expected.size());
  EXPECT_EQ(cudaMemcpy(got.data(), out_buf->data(), got.size() * sizeof(bf16),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  return got;
}

TEST_F(Mxfp4Test, DecodesEveryGoldenCaseExactly) {
  for (const GoldenCase& gc : cases_) {
    const std::vector<bf16> got = DecodeOnDevice(gc, /*gate_up=*/false);

    size_t mismatches = 0;
    size_t first = 0;

    for (size_t i = 0; i < gc.expected.size(); ++i) {
      if (__bfloat162float(got[i]) != gc.expected[i]) {
        if (mismatches == 0) first = i;
        ++mismatches;
      }
    }

    EXPECT_EQ(mismatches, 0u)
        << gc.label << ": " << mismatches << " of " << gc.expected.size()
        << " values differ; first at " << first << " (got "
        << __bfloat162float(got[first]) << ", want " << gc.expected[first]
        << "). A swapped nibble order would put this near half of them.";
  }
}

TEST_F(Mxfp4Test, GoldenCasesAreNotDegenerate) {
  // A decoder that returned all zeros would pass an exactness check against an
  // all-zero expectation. MXFP4 has a genuine zero, so some are expected --
  // but a weight slice that is mostly zero means the golden file is wrong, not
  // that the model is.
  for (const GoldenCase& gc : cases_) {
    size_t nonzero = 0;
    for (float v : gc.expected) {
      if (v != 0.0f) ++nonzero;
    }

    EXPECT_GT(nonzero, gc.expected.size() / 2)
        << gc.label << " is mostly zeros (" << nonzero << " of "
        << gc.expected.size() << " nonzero)";
  }
}

TEST_F(Mxfp4Test, EveryDecodedValueSurvivesBf16Exactly) {
  // The claim mxfp4.h makes, checked rather than asserted: an FP4 value scaled
  // by a power of two is exactly representable in bf16, so the decode loses
  // nothing and the test above is entitled to demand equality.
  for (const GoldenCase& gc : cases_) {
    for (size_t i = 0; i < gc.expected.size(); ++i) {
      const float v = gc.expected[i];
      ASSERT_EQ(__bfloat162float(__float2bfloat16(v)), v)
          << gc.label << ": value " << v << " at " << i
          << " does not round-trip through bf16, so the format's exactness "
             "claim is wrong";
    }
  }
}

TEST_F(Mxfp4Test, GateUpVariantPermutesRowsAndNothingElse) {
  // The de-interleave is a row permutation and must change nothing else: row
  // 2i of the source becomes row i of the output, row 2i+1 becomes row
  // half + i. Checked against the plain decode of the same bytes, so a bug in
  // the decode itself cannot hide here.
  for (const GoldenCase& gc : cases_) {
    if (gc.rows % 2 != 0) continue;

    const std::vector<bf16> plain = DecodeOnDevice(gc, /*gate_up=*/false);
    const std::vector<bf16> split = DecodeOnDevice(gc, /*gate_up=*/true);

    const int64_t width = gc.num_blocks * kValuesPerBlock;
    const int64_t half = gc.rows / 2;

    for (int64_t r = 0; r < gc.rows; ++r) {
      const int64_t dest = (r % 2 == 0) ? r / 2 : half + r / 2;

      for (int64_t c = 0; c < width; ++c) {
        ASSERT_EQ(__bfloat162float(split[static_cast<size_t>(dest * width + c)]),
                  __bfloat162float(plain[static_cast<size_t>(r * width + c)]))
            << gc.label << ": source row " << r << " did not land at " << dest;
      }
    }
  }
}

// The other half of Phase 1: that the loader can actually read these tensors.
//
// The golden file's cases are the first 32 rows of named experts, so the whole
// chain -- checkpoint on disk, our safetensors reader, our decode kernel -- can
// be checked against HuggingFace's answer without the golden file and the
// checkpoint ever having met. If the loader mis-strides a u8 rank-4 tensor,
// this fails where the test above passes.
TEST_F(Mxfp4Test, LoadsAndDecodesStraightFromTheCheckpoint) {
  const char* dir = std::getenv("INFERX_TEST_GPTOSS_CHECKPOINT");
  std::string path = dir != nullptr ? dir : "";

  if (path.empty()) {
    if (const char* home = std::getenv("HOME")) {
      path = std::string(home) +
             "/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
             "6cee5e81ee83917806bbde320786a8fb61efebee";
    }
  }

  auto ckpt = model::Checkpoint::Open(path);
  if (!ckpt.ok()) GTEST_SKIP() << "no gpt-oss checkpoint at " << path;

  // Case labels encode where they came from; map them back to tensor names.
  const struct {
    const char* label;
    const char* base;
    int64_t expert;
  } kSources[] = {
      {"layer0.gate_up.expert0", "model.layers.0.mlp.experts.gate_up_proj", 0},
      {"layer0.down.expert0", "model.layers.0.mlp.experts.down_proj", 0},
      {"layer23.gate_up.expert31", "model.layers.23.mlp.experts.gate_up_proj", 31},
  };

  for (const GoldenCase& gc : cases_) {
    const auto* source = std::find_if(
        std::begin(kSources), std::end(kSources),
        [&](const auto& s) { return gc.label == s.label; });
    ASSERT_NE(source, std::end(kSources)) << "unknown case " << gc.label;

    auto blocks = ckpt->Get(std::string(source->base) + "_blocks");
    auto scales = ckpt->Get(std::string(source->base) + "_scales");
    auto bias = ckpt->Get(std::string(source->base) + "_bias");
    ASSERT_TRUE(blocks.ok()) << blocks.status();
    ASSERT_TRUE(scales.ok()) << scales.status();
    ASSERT_TRUE(bias.ok()) << bias.status();

    EXPECT_EQ(blocks->dtype(), DataType::kUInt8);
    EXPECT_EQ(scales->dtype(), DataType::kUInt8);
    EXPECT_EQ(bias->dtype(), DataType::kBFloat16);
    ASSERT_EQ(blocks->rank(), 4) << "[experts, rows, blocks, 16]";
    EXPECT_EQ(blocks->dim(3), kBytesPerBlock);
    EXPECT_EQ(blocks->dim(1), scales->dim(1));

    // The rows the golden file covers, for this expert.
    const int64_t rows = gc.rows;
    const int64_t nb = gc.num_blocks;
    ASSERT_EQ(nb, blocks->dim(2));

    const auto* block_base = static_cast<const uint8_t*>(blocks->data());
    const auto* scale_base = static_cast<const uint8_t*>(scales->data());

    const size_t block_stride =
        static_cast<size_t>(blocks->dim(1)) * nb * kBytesPerBlock;
    const size_t scale_stride = static_cast<size_t>(scales->dim(1)) * nb;

    std::vector<uint8_t> host_blocks(
        block_base + source->expert * block_stride,
        block_base + source->expert * block_stride +
            static_cast<size_t>(rows) * nb * kBytesPerBlock);
    std::vector<uint8_t> host_scales(
        scale_base + source->expert * scale_stride,
        scale_base + source->expert * scale_stride +
            static_cast<size_t>(rows) * nb);

    // Same bytes the generator saw? If not, the loader is the problem and the
    // decode comparison below would be blaming the wrong component.
    ASSERT_EQ(host_blocks, gc.blocks) << gc.label << ": loader read different "
                                         "bytes than the golden generator";
    ASSERT_EQ(host_scales, gc.scales) << gc.label << ": scales differ";

    GoldenCase from_disk = gc;
    from_disk.blocks = std::move(host_blocks);
    from_disk.scales = std::move(host_scales);

    const std::vector<bf16> got = DecodeOnDevice(from_disk, /*gate_up=*/false);
    for (size_t i = 0; i < gc.expected.size(); ++i) {
      ASSERT_EQ(__bfloat162float(got[i]), gc.expected[i])
          << gc.label << " differs at " << i;
    }
  }
}

TEST_F(Mxfp4Test, RejectsAnOddRowCountForGateUp) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  // gate and up interleave in pairs, so an odd row count is a malformed weight
  // rather than something to round down.
  constexpr int64_t rows = 3, num_blocks = 1;

  auto blocks_buf = DeviceBuffer::Allocate(rows * num_blocks * kBytesPerBlock,
                                           DeviceId::Cuda(0));
  auto scales_buf = DeviceBuffer::Allocate(rows * num_blocks, DeviceId::Cuda(0));
  auto out_buf = DeviceBuffer::Allocate(
      rows * num_blocks * kValuesPerBlock * sizeof(bf16), DeviceId::Cuda(0));
  ASSERT_TRUE(blocks_buf.ok() && scales_buf.ok() && out_buf.ok());

  auto blocks = TensorView::Create(blocks_buf->data(), DataType::kUInt8,
                                   Shape({rows, num_blocks, kBytesPerBlock}),
                                   DeviceId::Cuda(0));
  auto scales = TensorView::Create(scales_buf->data(), DataType::kUInt8,
                                   Shape({rows, num_blocks}), DeviceId::Cuda(0));
  auto out = TensorView::Create(out_buf->data(), DataType::kBFloat16,
                                Shape({rows, num_blocks * kValuesPerBlock}),
                                DeviceId::Cuda(0));
  ASSERT_TRUE(blocks.ok() && scales.ok() && out.ok());

  EXPECT_TRUE(kernels::DequantizeMxfp4ToBf16(*blocks, *scales, *out).ok());
  EXPECT_FALSE(
      kernels::DequantizeMxfp4GateUpToBf16(*blocks, *scales, *out).ok());
}

TEST(Mxfp4GroupedGemmTest, MatchesAHostReferenceForRaggedExperts) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  // Expert 1 is empty; experts 0 and 2 both cross the kernel's 16-row chunk
  // boundary. Six output rows also exercise gate/up de-interleaving.
  constexpr int64_t experts = 3, assignments = 35, n = 6, k = 32;
  const std::vector<int32_t> offsets = {0, 17, 17, 35};
  std::vector<bf16> x(assignments * k);
  std::vector<uint8_t> blocks(experts * n * k / 2);
  std::vector<uint8_t> scales(experts * n * k / 32, 127);  // 2^(127-127) = 1
  std::vector<bf16> bias(experts * n);

  for (int64_t i = 0; i < assignments * k; ++i)
    x[static_cast<size_t>(i)] =
        __float2bfloat16(static_cast<float>((i * 7) % 13 - 6) / 8.0f);
  for (int64_t i = 0; i < experts * n * k / 2; ++i) {
    const uint8_t lo = static_cast<uint8_t>((i * 3 + 1) & 0xf);
    const uint8_t hi = static_cast<uint8_t>((i * 5 + 2) & 0xf);
    blocks[static_cast<size_t>(i)] = static_cast<uint8_t>(lo | (hi << 4));
  }
  for (int64_t e = 0; e < experts; ++e)
    for (int64_t j = 0; j < n; ++j)
      bias[static_cast<size_t>(e * n + j)] =
          __float2bfloat16(static_cast<float>(10 * e + j) / 16.0f);

  auto xb = DeviceBuffer::Allocate(x.size() * sizeof(bf16), DeviceId::Cuda(0));
  auto ob = DeviceBuffer::Allocate(offsets.size() * sizeof(int32_t),
                                   DeviceId::Cuda(0));
  auto wb = DeviceBuffer::Allocate(blocks.size(), DeviceId::Cuda(0));
  auto sb = DeviceBuffer::Allocate(scales.size(), DeviceId::Cuda(0));
  auto bb =
      DeviceBuffer::Allocate(bias.size() * sizeof(bf16), DeviceId::Cuda(0));
  auto yb =
      DeviceBuffer::Allocate(assignments * n * sizeof(bf16), DeviceId::Cuda(0));
  ASSERT_TRUE(xb.ok() && ob.ok() && wb.ok() && sb.ok() && bb.ok() && yb.ok());
  ASSERT_EQ(
      cudaMemcpy(xb->data(), x.data(), xb->size(), cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(cudaMemcpy(ob->data(), offsets.data(), ob->size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(wb->data(), blocks.data(), wb->size(), cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(sb->data(), scales.data(), sb->size(), cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(
      cudaMemcpy(bb->data(), bias.data(), bb->size(), cudaMemcpyHostToDevice),
      cudaSuccess);
  ASSERT_EQ(cudaMemset(yb->data(), 0x7f, yb->size()), cudaSuccess);

  auto xv = TensorView::Create(xb->data(), DataType::kBFloat16,
                               Shape({assignments, k}), DeviceId::Cuda(0));
  auto ov = TensorView::Create(ob->data(), DataType::kInt32,
                               Shape({experts + 1}), DeviceId::Cuda(0));
  auto wv = TensorView::Create(wb->data(), DataType::kUInt8,
                               Shape({experts, n, k / 32, 16}),
                               DeviceId::Cuda(0));
  auto sv = TensorView::Create(sb->data(), DataType::kUInt8,
                               Shape({experts, n, k / 32}), DeviceId::Cuda(0));
  auto bv = TensorView::Create(bb->data(), DataType::kBFloat16,
                               Shape({experts, n}), DeviceId::Cuda(0));
  auto yv = TensorView::Create(yb->data(), DataType::kBFloat16,
                               Shape({assignments, n}), DeviceId::Cuda(0));
  ASSERT_TRUE(xv.ok() && ov.ok() && wv.ok() && sv.ok() && bv.ok() && yv.ok());
  ASSERT_TRUE(kernels::Mxfp4GroupedGemm(*xv, *ov, *wv, *sv, *bv, *yv,
                                        /*deinterleave=*/true)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);

  std::vector<bf16> got(assignments * n);
  ASSERT_EQ(
      cudaMemcpy(got.data(), yb->data(), yb->size(), cudaMemcpyDeviceToHost),
      cudaSuccess);
  constexpr float lut[16] = {0, 0.5f,  1,  1.5f,  2,  3,  4,  6,
                             0, -0.5f, -1, -1.5f, -2, -3, -4, -6};
  for (int64_t e : {0, 2}) {
    for (int64_t row = offsets[e]; row < offsets[e + 1]; ++row) {
      for (int64_t j_raw = 0; j_raw < n; ++j_raw) {
        const int64_t j = (j_raw & 1) ? n / 2 + j_raw / 2 : j_raw / 2;
        float want = __bfloat162float(bias[static_cast<size_t>(e * n + j)]);
        const int64_t weight_row = e * n + j_raw;
        for (int64_t kk = 0; kk < k; ++kk) {
          const uint8_t packed =
              blocks[static_cast<size_t>(weight_row * (k / 2) + kk / 2)];
          const uint8_t nibble = (kk & 1) ? packed >> 4 : packed & 0xf;
          want += __bfloat162float(x[static_cast<size_t>(row * k + kk)]) *
                  lut[nibble];
        }
        EXPECT_EQ(__bfloat162float(got[static_cast<size_t>(row * n + j)]),
                  __bfloat162float(__float2bfloat16(want)))
            << "expert=" << e << " row=" << row << " j_raw=" << j_raw;
      }
    }
  }

  // Exercise the decode specialization explicitly. Four grouped assignments
  // take the direct grouped-row -> expert binary lookup; expert 1 is empty and
  // the two active experts own two rows each, so this also checks boundaries
  // rather than only the usual one-row-per-selected-expert case.
  const std::vector<int32_t> decode_offsets = {0, 2, 2, 4};
  ASSERT_EQ(cudaMemcpy(ob->data(), decode_offsets.data(), ob->size(),
                       cudaMemcpyHostToDevice),
            cudaSuccess);
  ASSERT_EQ(cudaMemset(yb->data(), 0x7f, 4 * n * sizeof(bf16)), cudaSuccess);
  auto x4 = TensorView::Create(xb->data(), DataType::kBFloat16, Shape({4, k}),
                               DeviceId::Cuda(0));
  auto y4 = TensorView::Create(yb->data(), DataType::kBFloat16, Shape({4, n}),
                               DeviceId::Cuda(0));
  ASSERT_TRUE(x4.ok() && y4.ok());
  ASSERT_TRUE(kernels::Mxfp4GroupedGemm(*x4, *ov, *wv, *sv, *bv, *y4,
                                        /*deinterleave=*/true)
                  .ok());
  ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
  std::vector<bf16> decode_got(4 * n);
  ASSERT_EQ(cudaMemcpy(decode_got.data(), yb->data(),
                       decode_got.size() * sizeof(bf16),
                       cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int64_t e : {0, 2}) {
    for (int64_t row = decode_offsets[e]; row < decode_offsets[e + 1]; ++row) {
      for (int64_t j_raw = 0; j_raw < n; ++j_raw) {
        const int64_t j = (j_raw & 1) ? n / 2 + j_raw / 2 : j_raw / 2;
        float want = __bfloat162float(bias[static_cast<size_t>(e * n + j)]);
        const int64_t weight_row = e * n + j_raw;
        for (int64_t kk = 0; kk < k; ++kk) {
          const uint8_t packed =
              blocks[static_cast<size_t>(weight_row * (k / 2) + kk / 2)];
          const uint8_t nibble = (kk & 1) ? packed >> 4 : packed & 0xf;
          want += __bfloat162float(x[static_cast<size_t>(row * k + kk)]) *
                  lut[nibble];
        }
        EXPECT_EQ(
            __bfloat162float(decode_got[static_cast<size_t>(row * n + j)]),
            __bfloat162float(__float2bfloat16(want)))
            << "decode expert=" << e << " row=" << row
            << " j_raw=" << j_raw;
      }
    }
  }
}

}  // namespace
}  // namespace inferx
