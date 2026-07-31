// Tests for the owning layer: Storage, TensorImpl, and the Tensor handle.
// TensorView (the non-owning POD) is covered in tensor_view_test.cc.

#include "inferx/core/tensor.h"

#include <cstring>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/caching_allocator.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/core/device_buffer.h"

namespace inferx {
namespace {

// ---------------------------------------------------------------------------
// Storage
// ---------------------------------------------------------------------------

TEST(Storage, AllocateOwnsAndFrees) {
  auto s = Storage::Allocate(4096, DeviceId::Cpu());
  ASSERT_TRUE(s.ok()) << s.status();
  EXPECT_EQ((*s)->Size(), 4096u);
  EXPECT_EQ((*s)->Device(), DeviceId::Cpu());
  EXPECT_FALSE((*s)->IsBorrowed());
  EXPECT_NE((*s)->Data(), nullptr);

  std::memset((*s)->Data(), 0x5A, (*s)->Size());
  EXPECT_EQ(static_cast<unsigned char>((*s)->Data()[4095]), 0x5Au);
}

TEST(Storage, BorrowFreesNothing) {
  std::vector<std::byte> owned(1024);
  {
    auto s = Storage::Borrow(owned.data(), owned.size(), DeviceId::Cpu());
    EXPECT_TRUE(s->IsBorrowed());
    EXPECT_EQ(s->Data(), owned.data());
  }
  // If Borrow had freed the vector's buffer this would be a use-after-free.
  owned[0] = std::byte{1};
  EXPECT_EQ(owned.size(), 1024u);
}

TEST(Storage, RejectsNullAllocator) {
  EXPECT_EQ(Storage::Allocate(16, nullptr).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(Storage, IsShared) {
  auto s = Storage::Allocate(256, DeviceId::Cpu());
  ASSERT_TRUE(s.ok());
  EXPECT_EQ((*s).UseCount(), 1u);
  {
    StoragePtr second = *s;
    EXPECT_EQ((*s).UseCount(), 2u);
  }
  EXPECT_EQ((*s).UseCount(), 1u);
}

// ---------------------------------------------------------------------------
// Tensor construction
// ---------------------------------------------------------------------------

TEST(Tensor, DefaultIsUndefined) {
  Tensor t;
  EXPECT_FALSE(t.defined());
  EXPECT_EQ(t.dtype(), DataType::kUndefined);
  EXPECT_EQ(t.numel(), 0);
  EXPECT_EQ(t.data(), nullptr);
  EXPECT_EQ(t.ToString(), "Tensor(<undefined>)");
}

// Accessors on an undefined tensor must not dereference null.
TEST(Tensor, UndefinedAccessorsAreSafe) {
  Tensor t;
  EXPECT_EQ(t.rank(), 0);
  EXPECT_EQ(t.nbytes(), 0);
  EXPECT_EQ(t.dim(0), 0);
  EXPECT_EQ(t.shape(), Shape{});
  EXPECT_EQ(t.use_count(), 0u);
  EXPECT_EQ(t.storage_use_count(), 0u);
  EXPECT_FALSE(t.view().IsDefined());
  EXPECT_FALSE(t.is_alias_of(Tensor()));
  EXPECT_EQ(t.Slice(0, 1).status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(t.Reshape(Shape{1}).status().code(),
            absl::StatusCode::kFailedPrecondition);
  EXPECT_EQ(t.Bitcast(DataType::kUInt8).status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST(Tensor, Empty) {
  auto t = Tensor::Empty(DataType::kBFloat16, Shape{4, 128}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok()) << t.status();
  EXPECT_TRUE(t->defined());
  EXPECT_EQ(t->numel(), 512);
  EXPECT_EQ(t->nbytes(), 1024);
  EXPECT_EQ(t->dtype(), DataType::kBFloat16);
  EXPECT_TRUE(t->is_cpu());
  EXPECT_EQ(t->use_count(), 1u);
  EXPECT_EQ(t->storage_use_count(), 1u);
  EXPECT_FALSE(t->storage()->IsBorrowed());
}

TEST(Tensor, EmptyValidatesLayout) {
  EXPECT_EQ(Tensor::Empty(DataType::kUndefined, Shape{4}, DeviceId::Cpu())
                .status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Tensor::Empty(DataType::kFloat, Shape{4, -1}, DeviceId::Cpu())
                .status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Tensor::Empty(DataType::kInt4, Shape{8, 65}, DeviceId::Cpu())
                .status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(Tensor::Empty(DataType::kFloat, Shape{1, 2, 3, 4, 5, 6, 7, 8, 9},
                          DeviceId::Cpu()).status().code(),
            absl::StatusCode::kInvalidArgument);
}

// The weight-loading path: one big allocation, many borrowed tensors over it.
TEST(Tensor, FromBlobBorrows) {
  std::vector<float> owned(256, 1.5f);
  auto t = Tensor::FromBlob(owned.data(), DataType::kFloat, Shape{16, 16},
                            DeviceId::Cpu());
  ASSERT_TRUE(t.ok()) << t.status();
  EXPECT_TRUE(t->storage()->IsBorrowed());
  EXPECT_EQ(t->data(), owned.data());
  EXPECT_EQ(t->data_as<float>()[0], 1.5f);

  t->reset();
  EXPECT_EQ(owned[0], 1.5f);  // not freed
}

TEST(Tensor, FromStorageChecksBounds) {
  auto s = Storage::Allocate(1024, DeviceId::Cpu());
  ASSERT_TRUE(s.ok());

  auto ok = Tensor::FromStorage(*s, 512, DataType::kFloat, Shape{128});
  ASSERT_TRUE(ok.ok()) << ok.status();
  EXPECT_EQ(ok->storage_offset(), 512);
  EXPECT_EQ(ok->data(), (*s)->Data() + 512);

  // 512 + 4*256 = 1536 > 1024
  EXPECT_EQ(Tensor::FromStorage(*s, 512, DataType::kFloat, Shape{256})
                .status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(Tensor::FromStorage(*s, -8, DataType::kFloat, Shape{1}).status().code(),
            absl::StatusCode::kOutOfRange);
  EXPECT_EQ(Tensor::FromStorage(nullptr, 0, DataType::kFloat, Shape{1})
                .status().code(),
            absl::StatusCode::kInvalidArgument);
}

// A large offset plus a large size must not wrap into an apparently valid range.
TEST(Tensor, FromStorageDoesNotOverflow) {
  auto s = Storage::Allocate(1024, DeviceId::Cpu());
  ASSERT_TRUE(s.ok());
  EXPECT_EQ(Tensor::FromStorage(*s, (int64_t{1} << 62), DataType::kFloat, Shape{1})
                .status().code(),
            absl::StatusCode::kOutOfRange);
}

// ---------------------------------------------------------------------------
// Sharing -- the reason this layer exists
// ---------------------------------------------------------------------------

TEST(Tensor, CopyIsCheapAndShares) {
  auto t = Tensor::Empty(DataType::kFloat, Shape{64}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());
  t->data_as<float>()[0] = 3.25f;

  Tensor copy = *t;
  EXPECT_EQ(t->use_count(), 2u);
  EXPECT_EQ(copy.data(), t->data());
  EXPECT_TRUE(copy.is_alias_of(*t));

  // Same bytes, not a duplicate.
  copy.data_as<float>()[0] = 9.5f;
  EXPECT_EQ(t->data_as<float>()[0], 9.5f);
}

TEST(Tensor, StorageOutlivesTheOriginalHandle) {
  Tensor slice;
  {
    auto base = Tensor::Empty(DataType::kInt32, Shape{16, 8}, DeviceId::Cpu());
    ASSERT_TRUE(base.ok());
    for (int i = 0; i < 128; ++i) base->data_as<int32_t>()[i] = i;

    auto s = base->Slice(4, 6);
    ASSERT_TRUE(s.ok()) << s.status();
    slice = *s;
    EXPECT_EQ(slice.storage_use_count(), 2u);
  }
  // base is gone; the slice still owns the storage and the bytes are intact.
  EXPECT_EQ(slice.storage_use_count(), 1u);
  EXPECT_EQ(slice.data_as<int32_t>()[0], 32);
  EXPECT_EQ(slice.data_as<int32_t>()[15], 47);
}

TEST(Tensor, SliceSharesStorageAndOffsets) {
  auto t = Tensor::Empty(DataType::kFloat, Shape{10, 8}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());

  auto s = t->Slice(2, 5);
  ASSERT_TRUE(s.ok()) << s.status();
  EXPECT_EQ(s->shape(), (Shape{3, 8}));
  EXPECT_TRUE(s->is_alias_of(*t));
  EXPECT_EQ(s->storage_offset(), 2 * 8 * 4);
  EXPECT_EQ(s->data(), static_cast<std::byte*>(t->data()) + 2 * 8 * 4);
}

// Offsets must accumulate rather than reset against the storage base.
TEST(Tensor, NestedSlicesAccumulateOffsets) {
  auto t = Tensor::Empty(DataType::kInt32, Shape{100}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());
  for (int i = 0; i < 100; ++i) t->data_as<int32_t>()[i] = i;

  auto a = t->Slice(10, 60);
  ASSERT_TRUE(a.ok());
  auto b = a->Slice(5, 15);
  ASSERT_TRUE(b.ok()) << b.status();

  EXPECT_EQ(b->storage_offset(), 15 * 4);
  EXPECT_EQ(b->numel(), 10);
  EXPECT_EQ(b->data_as<int32_t>()[0], 15);
  EXPECT_EQ(b->data_as<int32_t>()[9], 24);
  EXPECT_TRUE(b->is_alias_of(*t));
}

TEST(Tensor, ReshapeAndBitcastShareStorage) {
  auto t = Tensor::Empty(DataType::kFloat, Shape{4, 8}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());

  auto r = t->Reshape(Shape{32});
  ASSERT_TRUE(r.ok()) << r.status();
  EXPECT_TRUE(r->is_alias_of(*t));
  EXPECT_EQ(r->data(), t->data());

  auto b = t->Bitcast(DataType::kUInt8);
  ASSERT_TRUE(b.ok()) << b.status();
  EXPECT_TRUE(b->is_alias_of(*t));
  EXPECT_EQ(b->numel(), 128);
  EXPECT_EQ(b->nbytes(), t->nbytes());
}

// A slice of a borrowed tensor stays borrowed; it must not acquire ownership of
// memory it never allocated.
TEST(Tensor, SliceOfBorrowedStaysBorrowed) {
  std::vector<int32_t> owned(64, 7);
  auto t = Tensor::FromBlob(owned.data(), DataType::kInt32, Shape{8, 8},
                            DeviceId::Cpu());
  ASSERT_TRUE(t.ok());
  auto s = t->Slice(2, 4);
  ASSERT_TRUE(s.ok());
  EXPECT_TRUE(s->storage()->IsBorrowed());
  EXPECT_EQ(s->data_as<int32_t>()[0], 7);
}

TEST(Tensor, DerivedViewsPropagateErrors) {
  auto t = Tensor::Empty(DataType::kFloat, Shape{10, 8}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());
  EXPECT_EQ(t->Slice(0, 11).status().code(), absl::StatusCode::kOutOfRange);
  EXPECT_EQ(t->Reshape(Shape{7}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(Tensor, IsAliasOfDistinguishesStorages) {
  auto a = Tensor::Empty(DataType::kFloat, Shape{8}, DeviceId::Cpu());
  auto b = Tensor::Empty(DataType::kFloat, Shape{8}, DeviceId::Cpu());
  ASSERT_TRUE(a.ok() && b.ok());
  EXPECT_FALSE(a->is_alias_of(*b));
  EXPECT_TRUE(a->is_alias_of(*a));
}

// ---------------------------------------------------------------------------
// Interop with the non-owning layer
// ---------------------------------------------------------------------------

TEST(Tensor, ViewMatchesTheOwningHandle) {
  auto t = Tensor::Empty(DataType::kBFloat16, Shape{4, 16}, DeviceId::Cpu());
  ASSERT_TRUE(t.ok());

  const TensorView v = t->view();
  EXPECT_EQ(v.Data(), t->data());
  EXPECT_EQ(v.GetDataType(), t->dtype());
  EXPECT_EQ(v.GetShape(), t->shape());
  EXPECT_EQ(v.Device(), t->device());
  EXPECT_EQ(v.NBytes(), t->nbytes());

  // Taking a view must not affect ownership -- it is not a handle.
  EXPECT_EQ(t->use_count(), 1u);
}

TEST(Tensor, HandleIsOnePointerWide) {
  static_assert(sizeof(Tensor) == sizeof(void*));
  SUCCEED();
}

// ---------------------------------------------------------------------------
// Allocator-backed ownership
// ---------------------------------------------------------------------------

TEST(Tensor, WorkspaceBackedTensorReturnsMemoryOnDestruction) {
  auto buf = DeviceBuffer::Allocate(1 << 20, DeviceId::Cpu());
  ASSERT_TRUE(buf.ok());
  CachingAllocator workspace(buf->data(), 1 << 20, DeviceId::Cpu());

  const size_t before = workspace.GetStats().in_use;
  {
    auto t = Tensor::Empty(DataType::kFloat, Shape{1024}, &workspace);
    ASSERT_TRUE(t.ok()) << t.status();
    EXPECT_EQ(workspace.GetStats().in_use, before + 4096);
    EXPECT_EQ(t->device(), DeviceId::Cpu());

    // Storage keeps the block alive while any handle survives.
    Tensor alias = *t;
    t->reset();
    EXPECT_EQ(workspace.GetStats().in_use, before + 4096);
  }
  EXPECT_EQ(workspace.GetStats().in_use, before);
  EXPECT_EQ(workspace.GetStats().num_deallocations, 1u);
}

// Alignment is a per-call parameter because the requirement belongs to the
// caller -- pinned memory needs a page, tensor data a cache line.
TEST(Allocator, HostHonoursRequestedAlignment) {
  auto a_or = AllocatorFor();
  ASSERT_TRUE(a_or.ok()) << a_or.status();
  Allocator* a = *a_or;
  for (size_t align : {size_t{16}, size_t{64}, kTensorAlignment, size_t{512},
                       kPageAlignment}) {
    auto p = a->Allocate(1000, align);
    ASSERT_TRUE(p.ok()) << "alignment " << align << ": " << p.status();
    EXPECT_EQ(reinterpret_cast<uintptr_t>(*p) % align, 0u)
        << "alignment " << align;
    EXPECT_TRUE(a->Deallocate(*p).ok());
  }
}

TEST(Allocator, DefaultAlignmentIsTensorAlignment) {
  auto a_or = AllocatorFor();
  ASSERT_TRUE(a_or.ok()) << a_or.status();
  Allocator* a = *a_or;
  auto p = a->Allocate(64);  // single-argument convenience
  ASSERT_TRUE(p.ok()) << p.status();
  EXPECT_EQ(reinterpret_cast<uintptr_t>(*p) % kTensorAlignment, 0u);
  EXPECT_TRUE(a->Deallocate(*p).ok());
}

TEST(Allocator, RejectsInvalidAlignment) {
  auto a_or = AllocatorFor();
  ASSERT_TRUE(a_or.ok()) << a_or.status();
  Allocator* a = *a_or;
  EXPECT_EQ(a->Allocate(64, 100).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(a->Allocate(64, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(Allocator, ZeroBytesIsOkAndNull) {
  auto a_or = AllocatorFor();
  ASSERT_TRUE(a_or.ok()) << a_or.status();
  Allocator* a = *a_or;
  auto p = a->Allocate(0);
  ASSERT_TRUE(p.ok()) << p.status();
  EXPECT_EQ(*p, nullptr);
  EXPECT_TRUE(a->Deallocate(nullptr).ok());
}

TEST(Allocator, HostSingletonIsStable) {
  auto a1 = AllocatorFor();
  auto a2 = AllocatorFor();
  ASSERT_TRUE(a1.ok()) << a1.status();
  ASSERT_TRUE(a2.ok()) << a2.status();
  EXPECT_EQ(*a1, *a2);
  EXPECT_EQ((*a1)->Device(), DeviceId::Cpu());
  EXPECT_EQ((*a1)->Name(), "host");
}

TEST(Allocator, DefaultForDeviceDispatches) {
  auto host = AllocatorFor(DeviceId::Cpu());
  ASSERT_TRUE(host.ok());
  auto defaulted = AllocatorFor();  // defaults to CPU
  ASSERT_TRUE(defaulted.ok()) << defaulted.status();
  EXPECT_EQ(*host, *defaulted);

  auto cuda = AllocatorFor(DeviceId::Cuda(0));
  if (kCudaEnabled) {
    ASSERT_TRUE(cuda.ok()) << cuda.status();
    EXPECT_EQ((*cuda)->Device(), DeviceId::Cuda(0));
  } else {
    EXPECT_EQ(cuda.status().code(), absl::StatusCode::kFailedPrecondition);
  }
}

TEST(Tensor, CudaTensorRoundTrip) {
  if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

  auto t = Tensor::Empty(DataType::kInt32, Shape{1024}, DeviceId::Cuda(0));
  ASSERT_TRUE(t.ok()) << t.status();
  EXPECT_TRUE(t->is_cuda());
  EXPECT_EQ(t->nbytes(), 4096);

#ifdef INFERX_WITH_CUDA
  std::vector<int32_t> src(1024), dst(1024, 0);
  for (int i = 0; i < 1024; ++i) src[i] = i * 3;
  ASSERT_EQ(cudaMemcpy(t->data(), src.data(), 4096, cudaMemcpyHostToDevice),
            cudaSuccess);

  // A slice must address the right bytes on the device too.
  auto s = t->Reshape(Shape{128, 8});
  ASSERT_TRUE(s.ok());
  auto rows = s->Slice(2, 4);
  ASSERT_TRUE(rows.ok());
  EXPECT_EQ(rows->storage_offset(), 2 * 8 * 4);

  std::vector<int32_t> got(16, 0);
  ASSERT_EQ(cudaMemcpy(got.data(), rows->data(), 64, cudaMemcpyDeviceToHost),
            cudaSuccess);
  for (int i = 0; i < 16; ++i) EXPECT_EQ(got[i], (16 + i) * 3);
#endif
}

}  // namespace
}  // namespace inferx
