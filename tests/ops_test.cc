#include <cstdint>
#include <limits>
#include "inferx/ops/sampling.h"
#include "inferx/ops/flash_attention.h"
#include <cstring>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/attention.h"
#include "inferx/ops/elementwise.h"
#include "inferx/ops/execution_context.h"
#include "inferx/ops/gather.h"
#include "inferx/ops/linear.h"
#include "inferx/ops/rotary.h"
#include "inferx/ops/rms_norm.h"

namespace inferx {
namespace {

class OpsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto runtime = RuntimeFor(DeviceId::Cuda(0));
    ASSERT_TRUE(runtime.ok());
    runtime_ = *runtime;
    ASSERT_TRUE(runtime_->Activate().ok());
    auto stream = runtime_->CreateStream();
    ASSERT_TRUE(stream.ok());
    stream_ = *stream;
  }

  Tensor MakeBf16(const Shape& shape) {
    auto tensor = Tensor::Empty(DataType::kBFloat16, shape, DeviceId::Cuda(0));
    EXPECT_TRUE(tensor.ok());
    return std::move(tensor).value();
  }

  Tensor Upload(const std::vector<float>& values, const Shape& shape) {
    Tensor tensor = MakeBf16(shape);
    std::vector<uint16_t> bf16(values.size());
    for (size_t i = 0; i < values.size(); ++i) bf16[i] = FloatToBf16Bits(values[i]);
    EXPECT_TRUE(runtime_
                    ->Copy(tensor.Data(), bf16.data(), bf16.size() * sizeof(uint16_t),
                           CopyKind::kHostToDevice)
                    .ok());
    return tensor;
  }

  Tensor UploadInt(const std::vector<int32_t>& values) {
    auto tensor = Tensor::Empty(DataType::kInt32, Shape({static_cast<int64_t>(values.size())}),
                                DeviceId::Cuda(0));
    EXPECT_TRUE(tensor.ok());
    Tensor out = std::move(tensor).value();
    EXPECT_TRUE(runtime_
                    ->Copy(out.Data(), values.data(), values.size() * sizeof(int32_t),
                           CopyKind::kHostToDevice)
                    .ok());
    return out;
  }

  std::vector<float> Download(const Tensor& tensor) {
    EXPECT_TRUE(runtime_->SynchronizeStream(stream_).ok());
    std::vector<uint16_t> bf16(tensor.Numel());
    EXPECT_TRUE(runtime_
                    ->Copy(bf16.data(), tensor.Data(), bf16.size() * sizeof(uint16_t),
                           CopyKind::kDeviceToHost)
                    .ok());
    std::vector<float> out(bf16.size());
    for (size_t i = 0; i < bf16.size(); ++i) {
      uint32_t bits = static_cast<uint32_t>(bf16[i]) << 16;
      std::memcpy(&out[i], &bits, sizeof(float));
    }
    return out;
  }

  static uint16_t FloatToBf16Bits(float value) {
    uint32_t bits;
    std::memcpy(&bits, &value, sizeof(bits));
    const uint32_t lsb = (bits >> 16) & 1;
    bits += 0x7fff + lsb;
    return static_cast<uint16_t>(bits >> 16);
  }

  DeviceRuntime* runtime_ = nullptr;
  Stream stream_;
};

TEST_F(OpsTest, GreedyArgmaxMatchesHostIncludingTiesAndNonFiniteValues) {
  constexpr int rows=5,vocab=151936,parts=(vocab+4095)/4096;
  std::vector<float> x(rows*vocab,-10.0f);
  x[4095]=x[4096]=x[151935]=3;
  x[vocab+8000]=7;
  x[2*vocab]=std::numeric_limits<float>::quiet_NaN(); x[2*vocab+1]=99;
  x[3*vocab+3]=std::numeric_limits<float>::quiet_NaN(); x[3*vocab+vocab-1]=99;
  std::fill(x.begin()+4*vocab,x.end(),-std::numeric_limits<float>::infinity());
  auto alloc=[&](DataType dtype,int64_t n) { return Tensor::Empty(dtype,Shape({n}),DeviceId::Cuda(0)).value(); };
  auto values=alloc(DataType::kFloat,rows*parts);
  auto indices=alloc(DataType::kInt32,rows*parts);
  auto output=alloc(DataType::kInt32,rows);
  ops::ExecutionContext ctx(*runtime_,stream_);
  for(auto dtype:{DataType::kFloat,DataType::kBFloat16}) {
    Tensor logits;
    if(dtype==DataType::kBFloat16) logits=Upload(x,Shape({rows,vocab}));
    else {
      logits=Tensor::Empty(dtype,Shape({rows,vocab}),DeviceId::Cuda(0)).value();
      ASSERT_TRUE(runtime_->Copy(logits.Data(),x.data(),x.size()*sizeof(float),CopyKind::kHostToDevice).ok());
    }
    ASSERT_TRUE(ops::GreedyArgmax(ctx,logits,values,indices,output).ok());
    ASSERT_TRUE(runtime_->SynchronizeStream(stream_).ok());
    std::vector<int> got(rows);
    ASSERT_TRUE(runtime_->Copy(got.data(),output.Data(),rows*sizeof(int),CopyKind::kDeviceToHost).ok());
    EXPECT_EQ(got,(std::vector<int>{4095,8000,0,vocab-1,0}));
  }
}

TEST_F(OpsTest, FlashAttentionMatchesDoubleReferenceWithRaggedPrefixAndShuffledPages) {
  constexpr int heads=4,kvheads=2,dim=128,page=16,blocks=12;
  const std::vector<int> lengths{129,33},kvptr{0,9,12};
  const std::vector<int> block_ids{11,10,9,8,7,6,5,4,3,2,1,0};
  const auto rounded=[&](float v) { uint32_t bits=uint32_t(FloatToBf16Bits(v))<<16; float out;std::memcpy(&out,&bits,4);return out; };
  std::vector<float> keys(blocks*page*kvheads*dim),values(keys.size());
  for(int s=0;s<2;++s) for(int pos=0;pos<lengths[s];++pos) for(int h=0;h<kvheads;++h) for(int d=0;d<dim;++d) {
    int index=((block_ids[kvptr[s]+pos/page]*page+pos%page)*kvheads+h)*dim+d;
    keys[index]=rounded(std::sin(float(pos*7+h*11+d*3+s))*0.7f);
    values[index]=rounded(std::cos(float(pos*3+h*5+d*7+s))*0.8f);
  }
  auto key=Upload(keys,Shape({blocks,page,kvheads,dim}));
  auto value=Upload(values,Shape({blocks,page,kvheads,dim}));
  auto kv=UploadInt(kvptr),ids=UploadInt(block_ids),last=UploadInt({1,1});
  ops::ExecutionContext ctx(*runtime_,stream_);
  for(int mode : {0, 1, 2, 3, 4}) {
    const bool decode = mode == 1 || mode == 2;
    std::vector<int> qo=decode?std::vector<int>{0,1,2}:std::vector<int>{0,2,5};
    std::vector<int> pos=decode?std::vector<int>{128,32}:std::vector<int>{127,128,30,31,32};
    if (mode >= 3) {
      qo = {0, 129, 162};
      pos.clear();
      for (int length : lengths) for (int i = 0; i < length; ++i) pos.push_back(i);
    }
    const int tile_rows = mode == 3 ? 128 : 64;
    int tiles = 0;
    for (int s = 0; s < 2; ++s) tiles += (2 * (qo[s + 1] - qo[s]) + tile_rows - 1) / tile_rows;
    const int tokens=pos.size();
    std::vector<float> queries(tokens*heads*dim);
    for(size_t i=0;i<queries.size();++i) queries[i]=rounded(std::sin(float(i*13))*0.9f);
    auto q=Upload(queries,Shape({tokens,heads*dim})),qo_d=UploadInt(qo);
    auto plan=UploadInt(std::vector<int>(3*tokens+1));
    auto out=MakeBf16(Shape({tokens,heads*dim}));
    ASSERT_TRUE(ops::PrepareFlashAttention(ctx,qo_d,plan,2,tiles,tile_rows).ok());
    ops::AttentionParams p; p.query_heads=heads;p.kv_heads=kvheads;p.head_dim=dim;p.scale=1/std::sqrt(float(dim));
    ops::FlashDecodeWorkspace workspace;
    if (mode == 2) {
      const int splits = 2 * ops::FlashDecodeWorkspace::kPartitions;
      workspace.plan = UploadInt(std::vector<int>(3 * splits + 4));
      workspace.values = MakeBf16(Shape({splits * heads * dim}));
      workspace.scores = Tensor::Empty(DataType::kFloat, Shape({splits * heads}),
                                       DeviceId::Cuda(0)).value();
      ASSERT_TRUE(ops::PrepareFlashDecode(ctx, kv, last, page, workspace).ok());
    }
    ASSERT_TRUE(ops::FlashPagedAttention(ctx,q,qo_d,kv,ids,last,key,value,page,p,plan,tiles,out,
                                        mode == 2 ? &workspace : nullptr,tile_rows).ok());
    const auto got=Download(out);
    for(int t=0;t<tokens;++t) for(int h=0;h<heads;++h) {
      const int s=t<qo[1]?0:1;
      std::vector<double> scores(pos[t]+1);
      for(int u=0;u<=pos[t];++u) {
        int index=((block_ids[kvptr[s]+u/page]*page+u%page)*kvheads+h/2)*dim;
        double dot=0;
        for(int d=0;d<dim;++d) dot+=double(queries[(t*heads+h)*dim+d])*keys[index+d];
        scores[u]=dot*p.scale;
      }
      double max=*std::max_element(scores.begin(),scores.end()),sum=0;
      for(auto& v:scores) {v=std::exp(v-max);sum+=v;}
      for(int d=0;d<dim;++d) {
        double expected=0;
        for(int u=0;u<=pos[t];++u) {
          int index=((block_ids[kvptr[s]+u/page]*page+u%page)*kvheads+h/2)*dim+d;
          expected+=scores[u]/sum*values[index];
        }
        EXPECT_NEAR(got[(t*heads+h)*dim+d],expected,0.005) << "decode="<<decode<<" token="<<t<<" head="<<h<<" dim="<<d;
      }
    }
  }
}

TEST_F(OpsTest, GatherRowsSelectsAndReorders) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  Tensor src = Upload({1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}, Shape({4, 3}));
  Tensor indices = UploadInt({3, 0, 3});
  Tensor out = MakeBf16(Shape({3, 3}));
  ASSERT_TRUE(ops::GatherRows(ctx, src, indices, out).ok());
  EXPECT_EQ(Download(out), (std::vector<float>{10, 11, 12, 1, 2, 3, 10, 11, 12}));
}

TEST_F(OpsTest, LinearMultipliesByWeightTranspose) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  Tensor x = Upload({1, 2, 3, 4, 5, 6}, Shape({2, 3}));
  Tensor weight = Upload({1, 0, -1, 0, 1, 0, 1, 1, 1, 2, -1, 0}, Shape({4, 3}));
  Tensor out = MakeBf16(Shape({2, 4}));
  ASSERT_TRUE(ops::Linear(ctx, x, weight, out).ok());
  // Row 0: [1-3, 2, 6, 0]; row 1: [4-6, 5, 15, 3].
  const auto got = Download(out);
  EXPECT_NEAR(got[0], -2, 0.1f);
  EXPECT_NEAR(got[1], 2, 0.05f);
  EXPECT_NEAR(got[2], 6, 0.05f);
  EXPECT_NEAR(got[3], 0, 0.05f);
  EXPECT_NEAR(got[4], -2, 0.1f);
  EXPECT_NEAR(got[5], 5, 0.1f);
  EXPECT_NEAR(got[6], 15, 0.1f);
  EXPECT_NEAR(got[7], 3, 0.1f);
}

TEST_F(OpsTest, DecodeLinearMatchesDoublePrecisionDotProducts) {
  const auto rounded = [&](float v) {
    uint32_t bits = uint32_t(FloatToBf16Bits(v)) << 16;
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
  };
  ops::ExecutionContext ctx(*runtime_, stream_);
  for (int batch : {1, 2, 3, 4, 5, 16}) for (int width : {256, 1024, 3072, 259}) {
    constexpr int channels = 37;  // Partial final output tile.
    std::vector<float> x(batch * width), w(channels * width);
    for (size_t i = 0; i < x.size(); ++i) x[i] = rounded(std::sin(float(i * 7)) * 0.1f);
    for (size_t i = 0; i < w.size(); ++i) w[i] = rounded(std::cos(float(i * 13)) * 0.1f);
    auto xd = Upload(x, Shape({batch, width}));
    auto wd = Upload(w, Shape({channels, width}));
    auto out = MakeBf16(Shape({batch, channels}));
    ASSERT_TRUE(ops::Linear(ctx, xd, wd, out).ok());
    const auto got = Download(out);
    for (int b = 0; b < batch; ++b) for (int c = 0; c < channels; ++c) {
      double expected = 0;
      for (int j = 0; j < width; ++j) expected += double(x[b * width + j]) * w[c * width + j];
      EXPECT_NEAR(got[b * channels + c], expected, 0.005)
          << "batch=" << batch << " width=" << width << " row=" << b << " channel=" << c;
    }
  }
}

TEST_F(OpsTest, FusedResidualNormExactlyMatchesSeparateRoundedOperations) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  for (int width : {128, 1024, 1031}) for (bool plus_one : {false, true}) {
    std::vector<float> x(7 * width), residual(x.size()), weight(width);
    for (size_t i = 0; i < x.size(); ++i) {
      x[i] = std::sin(float(i)) * 2;
      residual[i] = std::cos(float(i * 7));
    }
    for (int i = 0; i < width; ++i) weight[i] = std::sin(float(i * 3));
    auto xd = Upload(x, Shape({7, width}));
    auto separate = Upload(residual, Shape({7, width}));
    auto fused = Upload(residual, Shape({7, width}));
    auto wd = Upload(weight, Shape({width}));
    auto expected = MakeBf16(Shape({7, width}));
    auto got = MakeBf16(Shape({7, width}));
    ops::RMSNormConfig config{1e-6f, plus_one, true};
    ASSERT_TRUE(ops::Add(ctx, xd, separate, separate).ok());
    ASSERT_TRUE(ops::RmsNorm(ctx, separate, wd, expected, config).ok());
    ASSERT_TRUE(ops::AddRmsNorm(ctx, xd, fused, wd, got, config).ok());
    EXPECT_EQ(Download(fused), Download(separate));
    EXPECT_EQ(Download(got), Download(expected));
    EXPECT_FALSE(ops::AddRmsNorm(ctx, xd, fused, wd, fused, config).ok());
  }
}

TEST_F(OpsTest, PackedProjectionsPreserveColumnOrderAndSiluRounding) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  for (int rows : {1, 4, 17}) {
    constexpr int qw = 19, kw = 7, vw = 11, width = qw + kw + vw;
    std::vector<float> packed(rows * width), qh, kh, vh;
    for (size_t i = 0; i < packed.size(); ++i) packed[i] = float(int(i % 31) - 15) / 8;
    for (int r = 0; r < rows; ++r) {
      qh.insert(qh.end(), packed.begin() + r * width, packed.begin() + r * width + qw);
      kh.insert(kh.end(), packed.begin() + r * width + qw, packed.begin() + r * width + qw + kw);
      vh.insert(vh.end(), packed.begin() + r * width + qw + kw, packed.begin() + (r + 1) * width);
    }
    auto pd = Upload(packed, Shape({rows, width}));
    auto q = MakeBf16(Shape({rows, qw}));
    auto k = MakeBf16(Shape({rows, kw}));
    auto v = MakeBf16(Shape({rows, vw}));
    ASSERT_TRUE(ops::SplitQkv(ctx, pd, q, k, v).ok());
    EXPECT_EQ(Download(q), qh);
    EXPECT_EQ(Download(k), kh);
    EXPECT_EQ(Download(v), vh);
    std::vector<float> gu, gate, up;
    for (int r = 0; r < rows; ++r) {
      gu.insert(gu.end(), qh.begin() + r * qw, qh.begin() + (r + 1) * qw);
      gu.insert(gu.end(), qh.begin() + r * qw, qh.begin() + (r + 1) * qw);
    }
    auto gd = Upload(qh, Shape({rows, qw}));
    auto gud = Upload(gu, Shape({rows, 2 * qw}));
    auto expected = MakeBf16(Shape({rows, qw}));
    ASSERT_TRUE(ops::SiluAndMul(ctx, gd, gd, expected).ok());
    ASSERT_TRUE(ops::PackedSiluAndMul(ctx, gud, q).ok());
    EXPECT_EQ(Download(q), Download(expected));
    EXPECT_FALSE(ops::SplitQkv(ctx, pd, k, k, v).ok());
    EXPECT_FALSE(ops::PackedSiluAndMul(ctx, pd, q).ok());
  }
}

TEST_F(OpsTest, FusedNormRopeExactlyMatchesSeparateRoundedOperations) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  constexpr int tokens = 3, qheads = 4, kheads = 2, dim = 128;
  std::vector<float> qh(tokens * qheads * dim), kh(tokens * kheads * dim), qw(dim), kw(dim);
  for (size_t i = 0; i < qh.size(); ++i) qh[i] = std::sin(float(i * 3)) * 2;
  for (size_t i = 0; i < kh.size(); ++i) kh[i] = std::cos(float(i * 7));
  for (int i = 0; i < dim; ++i) {
    qw[i] = std::cos(float(i * 5));
    kw[i] = std::sin(float(i * 11));
  }
  auto qweight = Upload(qw, Shape({dim})), kweight = Upload(kw, Shape({dim}));
  auto positions = UploadInt({0, 17, 1024});
  for (int rotary : {64, 128}) {
    auto q = Upload(qh, Shape({tokens, qheads, dim}));
    auto k = Upload(kh, Shape({tokens, kheads, dim}));
    auto qref = Upload(qh, Shape({tokens, qheads, dim}));
    auto kref = Upload(kh, Shape({tokens, kheads, dim}));
    auto qrows = qref.Reshape(Shape({tokens * qheads, dim})).value();
    auto krows = kref.Reshape(Shape({tokens * kheads, dim})).value();
    ASSERT_TRUE(ops::RmsNorm(ctx, qrows, qweight, qrows, {1e-6f, false, true}).ok());
    ASSERT_TRUE(ops::RmsNorm(ctx, krows, kweight, krows, {1e-6f, false, true}).ok());
    ASSERT_TRUE(ops::ApplyRope(ctx, qref, kref, positions, {rotary, 1000000.0f}).ok());
    ASSERT_TRUE(ops::NormalizeAndApplyRope(ctx, q, k, qweight, kweight, positions,
                                          1e-6f, {rotary, 1000000.0f}).ok());
    EXPECT_EQ(Download(q), Download(qref));
    EXPECT_EQ(Download(k), Download(kref));
  }
}

TEST_F(OpsTest, VectorCacheWritePreservesShuffledPagesAndUntouchedSlots) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  // Cross the dispatch threshold, page boundaries, and both supported and
  // scalar-fallback widths. Interleave sequences with unequal cached prefixes.
  for (int tokens : {1, 31, 32, 33, 67, 512}) {
    for (int sequences : {1, 4, 16}) {
      for (int heads : {1, 2, 4, 8, 16}) {
        for (int dim : {7, 128}) {
          SCOPED_TRACE(::testing::Message() << "tokens=" << tokens << " batch=" << sequences
                                          << " heads=" << heads << " dim=" << dim);
          const int width = heads * dim, page = 16;
          const int pages_per_seq = ((tokens + sequences - 1) / sequences + 19 + page - 1) / page;
          const int blocks = sequences * pages_per_seq;
          std::vector<int> pages(blocks), indptr(sequences + 1);
          for (int i = 0; i < blocks; ++i) pages[i] = (i * (blocks - 1) + 1) % blocks;
          for (int i = 0; i <= sequences; ++i) indptr[i] = i * pages_per_seq;
          std::vector<float> keys(tokens * width), values(keys.size());
          std::vector<float> expected_k(blocks * page * width, -1), expected_v(expected_k);
          std::vector<int> positions(tokens), batches(tokens);
          for (int t = 0; t < tokens; ++t) {
            const int seq = t % sequences, pos = t / sequences + (seq * 7 + 3) % 20;
            positions[t] = pos; batches[t] = seq;
            const int dst = (pages[indptr[seq] + pos / page] * page + pos % page) * width;
            for (int j = 0; j < width; ++j) {
              keys[t * width + j] = float((t * 7 + j) % 31) / 8;
              values[t * width + j] = -keys[t * width + j];
              expected_k[dst + j] = keys[t * width + j];
              expected_v[dst + j] = values[t * width + j];
            }
          }
          auto k = Upload(keys, Shape({tokens, width})), v = Upload(values, Shape({tokens, width}));
          auto kc = Upload(std::vector<float>(expected_k.size(), -1), Shape({blocks, page, heads, dim}));
          auto vc = Upload(std::vector<float>(expected_v.size(), -1), Shape({blocks, page, heads, dim}));
          auto pos = UploadInt(positions), batch = UploadInt(batches), ptr = UploadInt(indptr), ids = UploadInt(pages);
          ASSERT_TRUE(ops::WritePagedKv(ctx, k, v, pos, batch, ptr, ids, kc, vc, page).ok());
          EXPECT_EQ(Download(kc), expected_k);
          EXPECT_EQ(Download(vc), expected_v);
        }
      }
    }
  }
}

TEST_F(OpsTest, AddAndSiluAndMulMatchReferences) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  Tensor a = Upload({1, -2, 0.5f, 3}, Shape({2, 2}));
  Tensor b = Upload({0.25f, 2, -0.5f, -1}, Shape({2, 2}));
  Tensor sum = MakeBf16(Shape({2, 2}));
  ASSERT_TRUE(ops::Add(ctx, a, b, sum).ok());
  const auto got_sum = Download(sum);
  EXPECT_NEAR(got_sum[0], 1.25f, 0.02f);
  EXPECT_NEAR(got_sum[1], 0, 0.02f);
  EXPECT_NEAR(got_sum[2], 0, 0.02f);
  EXPECT_NEAR(got_sum[3], 2, 0.02f);

  Tensor act = MakeBf16(Shape({2, 2}));
  ASSERT_TRUE(ops::SiluAndMul(ctx, a, b, act).ok());
  const auto got_act = Download(act);
  const std::vector<float> gate = {1, -2, 0.5f, 3}, up = {0.25f, 2, -0.5f, -1};
  for (int i = 0; i < 4; ++i) {
    const float silu = gate[i] / (1.0f + std::exp(-gate[i]));
    EXPECT_NEAR(got_act[i], silu * up[i], 0.05f) << "element " << i;
  }
}

TEST_F(OpsTest, ApplyRopeRotatesHalfPairs) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  // Two tokens, one query head and one kv head, head_dim 4, rotary_dim 4.
  const float theta = 10000.0f;
  Tensor q = Upload({1, 2, 3, 4, 1, 2, 3, 4}, Shape({2, 1, 4}));
  Tensor k = Upload({5, 6, 7, 8, 5, 6, 7, 8}, Shape({2, 1, 4}));
  Tensor positions = UploadInt({0, 2});
  ASSERT_TRUE(ops::ApplyRope(ctx, q, k, positions, ops::RotaryParams{4, theta}).ok());
  const auto got_q = Download(q);
  const auto got_k = Download(k);
  // inv_freq[i] = theta^(-2i/rotary_dim); pair 0 rotates by pos * 1 rad.
  const float angle0 = 2.0f;
  const float angle1 = 2.0f * std::pow(theta, -0.5f);
  const float c0 = std::cos(angle0), s0 = std::sin(angle0);
  const float c1 = std::cos(angle1), s1 = std::sin(angle1);
  // Token 0 passes through; token 2 rotates pairs (0,2) and (1,3).
  EXPECT_NEAR(got_q[0], 1, 0.02f);
  EXPECT_NEAR(got_q[4], 1 * c0 - 3 * s0, 0.02f);
  EXPECT_NEAR(got_q[5], 2 * c1 - 4 * s1, 0.02f);
  EXPECT_NEAR(got_q[6], 3 * c0 + 1 * s0, 0.02f);
  EXPECT_NEAR(got_q[7], 4 * c1 + 2 * s1, 0.02f);
  EXPECT_NEAR(got_k[5], 6 * c1 - 8 * s1, 0.02f);
}

TEST_F(OpsTest, ApplyRopeMatchesBf16ReferenceRounding) {
  // PyTorch BF16 fixture, including GQA, partial rotation, a page boundary
  // and a long position. Each product is rounded before the sum/difference.
  std::vector<float> queries(48), keys(24);
  for (int i = 0; i < 48; ++i) queries[i] = (i + 1) / 8.0f;
  for (int t = 0; t < 3; ++t)
    for (int d = 0; d < 8; ++d) keys[t * 8 + d] = -queries[t * 16 + d];
  Tensor q = Upload(queries, Shape({3, 2, 8}));
  Tensor k = Upload(keys, Shape({3, 1, 8}));
  Tensor positions = UploadInt({0, 17, 1024});
  ops::ExecutionContext ctx(*runtime_, stream_);
  ASSERT_TRUE(ops::ApplyRope(ctx, q, k, positions, ops::RotaryParams{4, 1e6f}).ok());
  const std::vector<float> expected_q{
      .125f, .25f, .375f, .5f, .625f, .75f, .875f, 1,
      1.125f, 1.25f, 1.375f, 1.5f, 1.625f, 1.75f, 1.875f, 2,
      1.6953125f, 2.203125f, -2.703125f, 2.53125f, 2.625f, 2.75f, 2.875f, 3,
      2.390625f, 3.1875f, -3.9375f, 3.5625f, 3.625f, 3.75f, 3.875f, 4,
      4.75f, -1.640625f, 3.65625f, 6, 4.625f, 4.75f, 4.875f, 5,
      5.90625f, -1.984375f, 4.5f, 7.375f, 5.625f, 5.75f, 5.875f, 6};
  std::vector<float> expected_k(24);
  for (int t = 0; t < 3; ++t)
    for (int d = 0; d < 8; ++d) expected_k[t * 8 + d] = -expected_q[t * 16 + d];
  EXPECT_EQ(Download(q), expected_q);
  EXPECT_EQ(Download(k), expected_k);
}

TEST_F(OpsTest, PagedAttentionMatchesCausalReference) {
  ops::ExecutionContext ctx(*runtime_, stream_);
  // Two sequences over a 3-block pool: seq 0 holds positions 0..2 (blocks 0
  // and 1), seq 1 holds positions 0..1 (block 2). One prefill-style call.
  constexpr int kTokens = 5, kQueryHeads = 4, kKvHeads = 2, kHeadDim = 64, kBlockSize = 2;
  constexpr int kNumBlocks = 3;
  // Deterministic small activations, indexed [token, head, dim] flattened.
  const auto value_at = [](int token, int head, int dim, float salt) {
    const int h = (token * 31 + head * 17 + dim * 7) % 23;
    return 0.09f * static_cast<float>(h) - 1.0f + salt;
  };
  std::vector<float> q_values(kTokens * kQueryHeads * kHeadDim);
  std::vector<float> k_values(kTokens * kKvHeads * kHeadDim);
  std::vector<float> v_values(kTokens * kKvHeads * kHeadDim);
  for (int t = 0; t < kTokens; ++t) {
    for (int h = 0; h < kQueryHeads; ++h) {
      for (int d = 0; d < kHeadDim; ++d) {
        q_values[(t * kQueryHeads + h) * kHeadDim + d] = value_at(t, h, d, 0.3f);
      }
    }
    for (int h = 0; h < kKvHeads; ++h) {
      for (int d = 0; d < kHeadDim; ++d) {
        k_values[(t * kKvHeads + h) * kHeadDim + d] = value_at(t, h, d, 0.0f);
        v_values[(t * kKvHeads + h) * kHeadDim + d] = value_at(t, h, d, 0.6f);
      }
    }
  }
  const std::vector<int32_t> positions = {0, 1, 2, 0, 1};
  const std::vector<int32_t> batch_indices = {0, 0, 0, 1, 1};
  const std::vector<int32_t> qo_indptr = {0, 3, 5};
  const std::vector<int32_t> kv_indptr = {0, 2, 3};
  const std::vector<int32_t> kv_indices = {0, 1, 2};

  Tensor q = Upload(q_values, Shape({kTokens, kQueryHeads * kHeadDim}));
  Tensor k = Upload(k_values, Shape({kTokens, kKvHeads * kHeadDim}));
  Tensor v = Upload(v_values, Shape({kTokens, kKvHeads * kHeadDim}));
  Tensor pos = UploadInt(positions);
  Tensor batch = UploadInt(batch_indices);
  Tensor qo = UploadInt(qo_indptr);
  Tensor kvp = UploadInt(kv_indptr);
  Tensor kvi = UploadInt(kv_indices);
  Tensor key_cache = MakeBf16(Shape({kNumBlocks, kBlockSize, kKvHeads, kHeadDim}));
  Tensor value_cache = MakeBf16(Shape({kNumBlocks, kBlockSize, kKvHeads, kHeadDim}));
  Tensor out = MakeBf16(Shape({kTokens, kQueryHeads * kHeadDim}));

  ASSERT_TRUE(ops::WritePagedKv(ctx, k, v, pos, batch, kvp, kvi, key_cache, value_cache,
                                kBlockSize)
                  .ok());
  ops::AttentionParams params;
  params.query_heads = kQueryHeads;
  params.kv_heads = kKvHeads;
  params.head_dim = kHeadDim;
  params.scale = 1.0f / std::sqrt(static_cast<float>(kHeadDim));
  Tensor last = UploadInt({1, 2});
  const int tiles = 2;
  Tensor plan = UploadInt(std::vector<int32_t>(3 * tiles + 1));
  ASSERT_TRUE(ops::PrepareFlashAttention(ctx, qo, plan, 2, tiles).ok());
  ASSERT_TRUE(ops::FlashPagedAttention(ctx, q, qo, kvp, kvi, last, key_cache, value_cache,
                                       kBlockSize, params, plan, tiles, out).ok());
  const auto got = Download(out);

  // Host reference: per (token, query head), causal softmax over the
  // sequence's cached keys of the shared kv head.
  const int group = kQueryHeads / kKvHeads;
  for (int t = 0; t < kTokens; ++t) {
    for (int h = 0; h < kQueryHeads; ++h) {
      const int kvh = h / group;
      const int seq = batch_indices[t];
      const int seq_begin = qo_indptr[seq];
      const int seq_end = qo_indptr[seq + 1];
      float max_score = -1e30f;
      std::vector<float> scores;
      for (int u = seq_begin; u < seq_end; ++u) {
        if (positions[u] > positions[t]) continue;
        float dot = 0.0f;
        for (int d = 0; d < kHeadDim; ++d) {
          dot += q_values[(t * kQueryHeads + h) * kHeadDim + d] *
                 k_values[(u * kKvHeads + kvh) * kHeadDim + d];
        }
        scores.push_back(dot * params.scale);
        max_score = std::max(max_score, scores.back());
      }
      float sum = 0.0f;
      std::vector<float> weights(scores.size());
      for (size_t i = 0; i < scores.size(); ++i) {
        weights[i] = std::exp(scores[i] - max_score);
        sum += weights[i];
      }
      std::vector<float> expected(kHeadDim, 0.0f);
      size_t idx = 0;
      for (int u = seq_begin; u < seq_end; ++u) {
        if (positions[u] > positions[t]) continue;
        for (int d = 0; d < kHeadDim; ++d) {
          expected[d] += weights[idx] / sum * v_values[(u * kKvHeads + kvh) * kHeadDim + d];
        }
        ++idx;
      }
      for (int d = 0; d < kHeadDim; ++d) {
        EXPECT_NEAR(got[(t * kQueryHeads + h) * kHeadDim + d], expected[d], 0.02f)
            << "token " << t << " head " << h << " dim " << d;
      }
    }
  }
}

}  // namespace
}  // namespace inferx
