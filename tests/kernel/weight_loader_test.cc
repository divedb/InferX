/// WeightLoader, the shared checkpoint→device movement layer.
///
/// The verbs are gathers — stack, permute, fuse — and a gather bug is a
/// silently wrong model, so every test here checks bytes, not just success.
/// The CPU-device mode runs the identical extent/segment/partition pipeline
/// without the pinned ring, which is what lets the transform logic be
/// verified byte-for-byte before the CUDA tests re-verify it through real
/// staging slots small enough that every path (slot split, stripe split,
/// dedicated chunk) is exercised on tensors of a few megabytes.

#include "inferx/model/weight_loader.h"

#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <unistd.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <numeric>
#include <optional>
#include <string>
#include <vector>

#include "absl/strings/str_cat.h"
#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/core/shape.h"
#include "inferx/model/parallel/partition.h"
#include "inferx/model/safetensors.h"

namespace inferx {
namespace {

using model::Checkpoint;
using model::WeightLoader;

// Writes a single-file fp32 safetensors checkpoint, the format the loader
// under test reads through Checkpoint.
class TempCheckpoint {
 public:
  TempCheckpoint() {
    char tmpl[] = "/tmp/inferx_wl_XXXXXX";
    dir_ = ::mkdtemp(tmpl) ? std::string(tmpl) : std::string();
  }

  ~TempCheckpoint() {
    if (dir_.empty()) return;
    ::unlink((dir_ + "/model.safetensors").c_str());
    ::rmdir(dir_.c_str());
  }

  const std::string& dir() const { return dir_; }

  void Add(const std::string& name, const std::vector<int64_t>& shape,
           std::vector<float> values) {
    entries_.push_back(Entry{name, shape, std::move(values)});
  }

  void Write() {
    std::string header = "{";
    size_t offset = 0;
    for (size_t i = 0; i < entries_.size(); ++i) {
      const Entry& e = entries_[i];
      const size_t bytes = e.data.size() * sizeof(float);
      absl::StrAppend(&header, i == 0 ? "" : ",", "\"", e.name,
                      "\":{\"dtype\":\"F32\",\"shape\":[");
      for (size_t d = 0; d < e.shape.size(); ++d) {
        absl::StrAppend(&header, d == 0 ? "" : ",", e.shape[d]);
      }
      absl::StrAppend(&header, "],\"data_offsets\":[", offset, ",",
                      offset + bytes, "]}");
      offset += bytes;
    }
    header += "}";

    std::ofstream out(dir_ + "/model.safetensors", std::ios::binary);
    const uint64_t header_len = header.size();
    out.write(reinterpret_cast<const char*>(&header_len), sizeof(header_len));
    out.write(header.data(), static_cast<std::streamsize>(header.size()));
    for (const Entry& e : entries_) {
      out.write(reinterpret_cast<const char*>(e.data.data()),
                static_cast<std::streamsize>(e.data.size() * sizeof(float)));
    }
  }

 private:
  struct Entry {
    std::string name;
    std::vector<int64_t> shape;
    std::vector<float> data;
  };

  std::string dir_;
  std::vector<Entry> entries_;
};

std::vector<float> Iota(size_t n, float start) {
  std::vector<float> v(n);
  std::iota(v.begin(), v.end(), start);
  return v;
}

// A fixture with small transform targets plus one tensor big enough to cross
// several staging slots and earn a dedicated device chunk under the shrunken
// options the CUDA tests use.
class WeightLoaderTest : public ::testing::Test {
 protected:
  static constexpr int64_t kBigRows = 1 << 18;
  static constexpr int64_t kBigCols = 8;  // 8 MB of fp32

  void SetUp() override {
    ckpt_.Add("a", {6, 4}, Iota(24, 0.0f));
    ckpt_.Add("p0", {2, 3}, Iota(6, 100.0f));
    ckpt_.Add("p1", {2, 3}, Iota(6, 200.0f));
    ckpt_.Add("p2", {2, 3}, Iota(6, 300.0f));
    ckpt_.Add("big", {kBigRows, kBigCols},
              Iota(static_cast<size_t>(kBigRows * kBigCols), 0.0f));
    ckpt_.Write();

    auto opened = Checkpoint::Open(ckpt_.dir());
    ASSERT_TRUE(opened.ok()) << opened.status().ToString();
    checkpoint_.emplace(std::move(opened).value());
  }

  // Small enough that "big" spans 8 slots and outgrows the chunk size.
  static WeightLoader::Options SmallOptions(DeviceId device) {
    WeightLoader::Options opts;
    opts.device = device;
    opts.staging_slot_bytes = size_t{1} << 20;
    opts.staging_slots = 2;
    opts.threads = 4;
    opts.device_chunk_bytes = size_t{4} << 20;
    return opts;
  }

  TempCheckpoint ckpt_;
  std::optional<Checkpoint> checkpoint_;
};

std::vector<float> ReadBack(const TensorView& view, int64_t numel,
                            DeviceId device) {
  std::vector<float> out(static_cast<size_t>(numel));
  if (device.IsCuda()) {
    EXPECT_EQ(cudaMemcpy(out.data(), view.Data(), out.size() * sizeof(float),
                         cudaMemcpyDeviceToHost),
              cudaSuccess);
  } else {
    std::memcpy(out.data(), view.Data(), out.size() * sizeof(float));
  }
  return out;
}

TEST_F(WeightLoaderTest, CpuLoadRoundTripsTheBytes) {
  auto loader =
      WeightLoader::Create(&*checkpoint_, SmallOptions(DeviceId::Cpu()));
  ASSERT_TRUE(loader.ok()) << loader.status().ToString();

  auto view = loader->Load("a", Shape({6, 4}));
  ASSERT_TRUE(view.ok()) << view.status().ToString();
  ASSERT_TRUE(loader->Finish().ok());

  EXPECT_EQ(ReadBack(*view, 24, DeviceId::Cpu()), Iota(24, 0.0f));
  EXPECT_EQ(loader->stats().tensors, 1u);
  EXPECT_EQ(loader->stats().bytes, 24 * sizeof(float));
}

TEST_F(WeightLoaderTest, CpuShardedLoadStreamsRowsAndColumns) {
  auto loader =
      WeightLoader::Create(&*checkpoint_, SmallOptions(DeviceId::Cpu()));
  ASSERT_TRUE(loader.ok()) << loader.status().ToString();

  auto rows = loader->Load(
      "a", Shape({6, 4}),
      model::parallel::ShardSpec{model::parallel::Partition::kRows, 1}, 1, 2);
  ASSERT_TRUE(rows.ok()) << rows.status().ToString();
  EXPECT_EQ(rows->Rank(), 2);
  EXPECT_EQ(rows->Dim(0), 3);
  EXPECT_EQ(rows->Dim(1), 4);
  EXPECT_EQ(ReadBack(*rows, 12, DeviceId::Cpu()), Iota(12, 12.0f));

  auto cols = loader->Load(
      "a", Shape({6, 4}),
      model::parallel::ShardSpec{model::parallel::Partition::kCols, 1}, 0, 2);
  ASSERT_TRUE(cols.ok()) << cols.status().ToString();
  EXPECT_EQ(cols->Rank(), 2);
  EXPECT_EQ(cols->Dim(0), 6);
  EXPECT_EQ(cols->Dim(1), 2);
  EXPECT_EQ(ReadBack(*cols, 12, DeviceId::Cpu()),
            (std::vector<float>{0, 1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21}));
}

TEST_F(WeightLoaderTest, CpuStackedConcatenatesInNameOrder) {
  auto loader =
      WeightLoader::Create(&*checkpoint_, SmallOptions(DeviceId::Cpu()));
  ASSERT_TRUE(loader.ok());

  const std::vector<std::string> names = {"p1", "p0", "p2"};
  auto view = loader->LoadStacked(names, Shape({2, 3}), Shape({3, 2, 3}));
  ASSERT_TRUE(view.ok()) << view.status().ToString();
  ASSERT_TRUE(loader->Finish().ok());

  std::vector<float> expected = Iota(6, 200.0f);
  const std::vector<float> mid = Iota(6, 100.0f);
  const std::vector<float> tail = Iota(6, 300.0f);
  expected.insert(expected.end(), mid.begin(), mid.end());
  expected.insert(expected.end(), tail.begin(), tail.end());

  EXPECT_EQ(ReadBack(*view, 18, DeviceId::Cpu()), expected);
}

TEST_F(WeightLoaderTest, CpuRowPermutedGathersRows) {
  auto loader =
      WeightLoader::Create(&*checkpoint_, SmallOptions(DeviceId::Cpu()));
  ASSERT_TRUE(loader.ok());

  const std::vector<int64_t> reversed = {5, 4, 3, 2, 1, 0};
  auto view = loader->LoadRowPermuted("a", Shape({6, 4}), reversed);
  ASSERT_TRUE(view.ok()) << view.status().ToString();
  ASSERT_TRUE(loader->Finish().ok());

  std::vector<float> expected;
  for (int64_t r = 5; r >= 0; --r) {
    for (int64_t c = 0; c < 4; ++c) {
      expected.push_back(static_cast<float>(r * 4 + c));
    }
  }
  EXPECT_EQ(ReadBack(*view, 24, DeviceId::Cpu()), expected);
}

TEST_F(WeightLoaderTest, CpuUploadTakesMaterializedHostTensors) {
  auto loader =
      WeightLoader::Create(&*checkpoint_, SmallOptions(DeviceId::Cpu()));
  ASSERT_TRUE(loader.ok());

  // A temporary the caller transforms itself — the TP-shard case. The loader
  // stages before returning, so the source may die right after the call.
  std::vector<float> scratch = Iota(8, 500.0f);
  {
    auto blob = Tensor::FromBlob(scratch.data(), DataType::kFloat,
                                 Shape({2, 4}), DeviceId::Cpu());
    ASSERT_TRUE(blob.ok());
    auto view = loader->Upload(*blob);
    ASSERT_TRUE(view.ok()) << view.status().ToString();
    ASSERT_TRUE(loader->Finish().ok());
    EXPECT_EQ(ReadBack(*view, 8, DeviceId::Cpu()), scratch);
  }

  // Heterogeneous stack: [6,4] and [2,3] parts fuse byte-wise into [30].
  auto a = checkpoint_->Get("a");
  auto p0 = checkpoint_->Get("p0");
  ASSERT_TRUE(a.ok() && p0.ok());
  const std::vector<Tensor> parts = {*a, *p0};
  auto stacked = loader->UploadStacked(parts, Shape({30}));
  ASSERT_TRUE(stacked.ok()) << stacked.status().ToString();
  ASSERT_TRUE(loader->Finish().ok());

  std::vector<float> expected = Iota(24, 0.0f);
  const std::vector<float> tail = Iota(6, 100.0f);
  expected.insert(expected.end(), tail.begin(), tail.end());
  EXPECT_EQ(ReadBack(*stacked, 30, DeviceId::Cpu()), expected);
}

TEST_F(WeightLoaderTest, UncheckedLoadUsesTheCheckpointShape) {
  auto loader =
      WeightLoader::Create(&*checkpoint_, SmallOptions(DeviceId::Cpu()));
  ASSERT_TRUE(loader.ok());

  auto view = loader->Load("a");
  ASSERT_TRUE(view.ok()) << view.status().ToString();
  EXPECT_EQ(view->Rank(), 2);
  EXPECT_EQ(view->Dim(0), 6);
  EXPECT_EQ(view->Dim(1), 4);
}

TEST_F(WeightLoaderTest, ZeroChunkBytesDedicatesOneBufferPerTensor) {
  WeightLoader::Options opts = SmallOptions(DeviceId::Cpu());
  opts.device_chunk_bytes = 0;
  auto loader = WeightLoader::Create(&*checkpoint_, opts);
  ASSERT_TRUE(loader.ok());

  ASSERT_TRUE(loader->Load("a", Shape({6, 4})).ok());
  EXPECT_EQ(loader->buffer_count(), 1u);
  ASSERT_TRUE(loader->Load("p0", Shape({2, 3})).ok());
  EXPECT_EQ(loader->buffer_count(), 2u);

  // One buffer per tensor, in load order; sizes may round up to alignment.
  auto buffers = loader->Release();
  ASSERT_TRUE(buffers.ok());
  ASSERT_EQ(buffers->size(), 2u);
  EXPECT_GE((*buffers)[0].size(), 24 * sizeof(float));
  EXPECT_GE((*buffers)[1].size(), 6 * sizeof(float));
}

TEST_F(WeightLoaderTest, ConcatenatedShapeSumsRows) {
  auto a = checkpoint_->Get("a");    // [6,4]
  auto p0 = checkpoint_->Get("p0");  // [2,3]
  ASSERT_TRUE(a.ok() && p0.ok());

  auto same = model::ConcatenatedShape({{*a, *a}});
  ASSERT_TRUE(same.ok());
  EXPECT_EQ(same->ToString(), Shape({12, 4}).ToString());

  // Mismatched widths refuse.
  EXPECT_FALSE(model::ConcatenatedShape({{*a, *p0}}).ok());
}

TEST_F(WeightLoaderTest, RejectsMismatchedInputs) {
  auto loader =
      WeightLoader::Create(&*checkpoint_, SmallOptions(DeviceId::Cpu()));
  ASSERT_TRUE(loader.ok());

  // Shape check comes from GetChecked and names the tensor.
  EXPECT_FALSE(loader->Load("a", Shape({4, 6})).ok());
  // Absent tensor.
  EXPECT_FALSE(loader->Load("nope", Shape({6, 4})).ok());
  // row_map must cover every row exactly.
  EXPECT_FALSE(loader->LoadRowPermuted("a", Shape({6, 4}), {{0, 1, 2}}).ok());
  // ...with in-range sources.
  EXPECT_FALSE(
      loader->LoadRowPermuted("a", Shape({6, 4}), {{0, 1, 2, 3, 4, 9}}).ok());
  // The declared out shape must match the parts exactly.
  const std::vector<std::string> names = {"p0", "p1"};
  EXPECT_FALSE(loader->LoadStacked(names, Shape({2, 3}), Shape({5, 3})).ok());
}

TEST_F(WeightLoaderTest, CudaMatchesTheHostGathers) {
  if (!cuda::Available()) GTEST_SKIP() << "no CUDA device";

  auto loader =
      WeightLoader::Create(&*checkpoint_, SmallOptions(DeviceId::Cuda(0)));
  ASSERT_TRUE(loader.ok()) << loader.status().ToString();

  // "big" (8 MB) crosses 8 staging slots and exceeds the 4 MB chunk size, so
  // it takes the dedicated-chunk path; the small tensors share a chunk.
  auto big = loader->Load("big", Shape({kBigRows, kBigCols}));
  ASSERT_TRUE(big.ok()) << big.status().ToString();

  const std::vector<std::string> names = {"p1", "p0", "p2"};
  auto stacked = loader->LoadStacked(names, Shape({2, 3}), Shape({3, 2, 3}));
  ASSERT_TRUE(stacked.ok()) << stacked.status().ToString();

  const std::vector<int64_t> reversed = {5, 4, 3, 2, 1, 0};
  auto permuted = loader->LoadRowPermuted("a", Shape({6, 4}), reversed);
  ASSERT_TRUE(permuted.ok()) << permuted.status().ToString();

  ASSERT_TRUE(loader->Finish().ok());

  const int64_t big_numel = kBigRows * kBigCols;
  EXPECT_EQ(ReadBack(*big, big_numel, DeviceId::Cuda(0)),
            Iota(static_cast<size_t>(big_numel), 0.0f));

  std::vector<float> stacked_expected = Iota(6, 200.0f);
  const std::vector<float> mid = Iota(6, 100.0f);
  const std::vector<float> tail = Iota(6, 300.0f);
  stacked_expected.insert(stacked_expected.end(), mid.begin(), mid.end());
  stacked_expected.insert(stacked_expected.end(), tail.begin(), tail.end());
  EXPECT_EQ(ReadBack(*stacked, 18, DeviceId::Cuda(0)), stacked_expected);

  std::vector<float> permuted_expected;
  for (int64_t r = 5; r >= 0; --r) {
    for (int64_t c = 0; c < 4; ++c) {
      permuted_expected.push_back(static_cast<float>(r * 4 + c));
    }
  }
  EXPECT_EQ(ReadBack(*permuted, 24, DeviceId::Cuda(0)), permuted_expected);

  EXPECT_EQ(loader->stats().tensors, 3u);
  EXPECT_EQ(loader->stats().bytes,
            static_cast<size_t>(big_numel + 18 + 24) * sizeof(float));

  // Release hands over every chunk backing the views; the caller keeps them.
  auto buffers = loader->Release();
  ASSERT_TRUE(buffers.ok());
  size_t reserved = 0;
  for (const DeviceBuffer& b : *buffers) reserved += b.size();
  EXPECT_GE(reserved, loader->stats().bytes);
}

}  // namespace
}  // namespace inferx
