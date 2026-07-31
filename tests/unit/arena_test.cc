#include "inferx/core/arena.h"

#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/device_buffer.h"

namespace inferx {
namespace {

class ArenaTest : public ::testing::Test {
 protected:
  void SetUp() override {
    auto buf = DeviceBuffer::Allocate(kCapacity, DeviceId::Cpu());
    ASSERT_TRUE(buf.ok()) << buf.status();
    buffer_ = *std::move(buf);
    arena_ = BumpArena(buffer_.data(), kCapacity, DeviceId::Cpu());
  }

  static constexpr size_t kCapacity = 1 << 20;
  DeviceBuffer buffer_;
  BumpArena arena_;
};

TEST_F(ArenaTest, AllocatesInOrder) {
  auto a = arena_.Allocate(1024);
  ASSERT_TRUE(a.ok()) << a.status();
  auto b = arena_.Allocate(1024);
  ASSERT_TRUE(b.ok()) << b.status();

  EXPECT_EQ(static_cast<std::byte*>(*b) - static_cast<std::byte*>(*a), 1024);
  EXPECT_EQ(arena_.Used(), 2048u);
  EXPECT_EQ(arena_.Remaining(), kCapacity - 2048);
}

TEST_F(ArenaTest, HonoursAlignment) {
  ASSERT_TRUE(arena_.Allocate(1).ok());
  auto p = arena_.Allocate(16, 256);
  ASSERT_TRUE(p.ok()) << p.status();
  EXPECT_EQ(reinterpret_cast<uintptr_t>(*p) % 256, 0u);

  auto q = arena_.Allocate(16, 4096);
  ASSERT_TRUE(q.ok()) << q.status();
  EXPECT_EQ(reinterpret_cast<uintptr_t>(*q) % 4096, 0u);
}

TEST_F(ArenaTest, RejectsNonPowerOfTwoAlignment) {
  EXPECT_EQ(arena_.Allocate(16, 100).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(arena_.Allocate(16, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST_F(ArenaTest, ExhaustionIsResourceExhausted) {
  ASSERT_TRUE(arena_.Allocate(kCapacity).ok());
  auto p = arena_.Allocate(1);
  EXPECT_EQ(p.status().code(), absl::StatusCode::kResourceExhausted);
  // The message needs to carry enough to size the arena correctly next time.
  EXPECT_NE(p.status().message().find("remaining"), std::string::npos);
}

// A size near SIZE_MAX must not wrap the offset computation into a "success".
TEST_F(ArenaTest, HugeRequestDoesNotOverflow) {
  auto p = arena_.Allocate(~size_t{0} - 128);
  EXPECT_EQ(p.status().code(), absl::StatusCode::kResourceExhausted);
  EXPECT_EQ(arena_.Used(), 0u);
}

TEST_F(ArenaTest, UninitializedArenaFails) {
  BumpArena empty;
  EXPECT_EQ(empty.Allocate(16).status().code(),
            absl::StatusCode::kFailedPrecondition);
}

TEST_F(ArenaTest, ResetRewindsButKeepsPeak) {
  ASSERT_TRUE(arena_.Allocate(4096).ok());
  EXPECT_EQ(arena_.PeakUsed(), 4096u);
  arena_.Reset();
  EXPECT_EQ(arena_.Used(), 0u);
  EXPECT_EQ(arena_.PeakUsed(), 4096u);

  ASSERT_TRUE(arena_.Allocate(16).ok());
  EXPECT_EQ(arena_.Used(), 16u);
  EXPECT_EQ(arena_.PeakUsed(), 4096u);
}

TEST_F(ArenaTest, AllocateView) {
  auto t = arena_.AllocateView(DataType::kBFloat16, Shape{32, 128});
  ASSERT_TRUE(t.ok()) << t.status();
  EXPECT_EQ(t->NBytes(), 32 * 128 * 2);
  EXPECT_EQ(t->Device(), DeviceId::Cpu());
  EXPECT_EQ(arena_.Used(), 32u * 128 * 2);
}

TEST_F(ArenaTest, AllocateViewPacksSubByte) {
  auto t = arena_.AllocateView(DataType::kInt4, Shape{64, 128});
  ASSERT_TRUE(t.ok()) << t.status();
  EXPECT_EQ(t->NBytes(), 64 * 128 / 2);
  EXPECT_EQ(arena_.Used(), 64u * 128 / 2);
}

TEST_F(ArenaTest, AllocateViewRejectsBadShape) {
  EXPECT_EQ(arena_.AllocateView(DataType::kFloat, Shape{4, -1}).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(arena_.AllocateView(DataType::kUndefined, Shape{4}).status().code(),
            absl::StatusCode::kInvalidArgument);
}

// Sanity check that the arena hands out genuinely usable, non-overlapping
// storage rather than plausible-looking pointers.
TEST_F(ArenaTest, RegionsDoNotOverlap) {
  auto a = arena_.AllocateView(DataType::kInt32, Shape{256});
  auto b = arena_.AllocateView(DataType::kInt32, Shape{256});
  ASSERT_TRUE(a.ok() && b.ok());

  int32_t* pa = a->DataAs<int32_t>();
  int32_t* pb = b->DataAs<int32_t>();
  ASSERT_NE(pa, nullptr);
  ASSERT_NE(pb, nullptr);

  for (int i = 0; i < 256; ++i) pa[i] = i;
  for (int i = 0; i < 256; ++i) pb[i] = -i;
  for (int i = 0; i < 256; ++i) {
    ASSERT_EQ(pa[i], i) << "at " << i;
    ASSERT_EQ(pb[i], -i) << "at " << i;
  }
}

}  // namespace
}  // namespace inferx
