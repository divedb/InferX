#include "inferx/core/intrusive_ref_cnt_ptr.h"

#include <atomic>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/storage.h"
#include "inferx/core/tensor.h"

namespace inferx {
namespace {

std::atomic<int> g_live{0};

class Thing : public ThreadSafeRefCountedBase<Thing> {
 public:
  explicit Thing(int v = 0) : value(v) { g_live.fetch_add(1); }
  ~Thing() { g_live.fetch_sub(1); }
  int value;
};

class IntrusiveRefCntPtrTest : public ::testing::Test {
 protected:
  void SetUp() override { g_live.store(0); }
  void TearDown() override { EXPECT_EQ(g_live.load(), 0) << "leaked objects"; }
};

TEST_F(IntrusiveRefCntPtrTest, IsOnePointerWide) {
  static_assert(sizeof(IntrusiveRefCntPtr<Thing>) == sizeof(void*));
  SUCCEED();
}

TEST_F(IntrusiveRefCntPtrTest, DefaultIsNull) {
  IntrusiveRefCntPtr<Thing> p;
  EXPECT_FALSE(p);
  EXPECT_EQ(p.Get(), nullptr);
  EXPECT_EQ(p.UseCount(), 0u);
  EXPECT_TRUE(p == nullptr);
}

TEST_F(IntrusiveRefCntPtrTest, MakeStartsAtOne) {
  auto p = makeIntrusiveRefCnt<Thing>(42);
  EXPECT_TRUE(p);
  EXPECT_EQ(p->value, 42);
  EXPECT_EQ(p.UseCount(), 1u);
  EXPECT_EQ(g_live.load(), 1);
}

TEST_F(IntrusiveRefCntPtrTest, DestroyedAtZero) {
  {
    auto p = makeIntrusiveRefCnt<Thing>();
    EXPECT_EQ(g_live.load(), 1);
  }
  EXPECT_EQ(g_live.load(), 0);
}

// Unlike a typical adopt-style constructor, this one *retains*. Constructing
// from a raw pointer therefore adds a reference rather than taking one over.
TEST_F(IntrusiveRefCntPtrTest, RawPointerConstructorRetains) {
  auto a = makeIntrusiveRefCnt<Thing>(11);
  IntrusiveRefCntPtr<Thing> b(a.Get());
  EXPECT_EQ(a.UseCount(), 2u);
  EXPECT_EQ(b->value, 11);
}

TEST_F(IntrusiveRefCntPtrTest, CopyShares) {
  auto a = makeIntrusiveRefCnt<Thing>(7);
  {
    auto b = a;
    EXPECT_EQ(a.UseCount(), 2u);
    EXPECT_EQ(a.Get(), b.Get());
  }
  EXPECT_EQ(a.UseCount(), 1u);
  EXPECT_EQ(g_live.load(), 1);
}

TEST_F(IntrusiveRefCntPtrTest, MoveDoesNotTouchTheCount) {
  auto a = makeIntrusiveRefCnt<Thing>(3);
  Thing* raw = a.Get();

  auto b = std::move(a);
  EXPECT_EQ(b.Get(), raw);
  EXPECT_EQ(b.UseCount(), 1u);
  EXPECT_FALSE(a);  // NOLINT(bugprone-use-after-move)
}

TEST_F(IntrusiveRefCntPtrTest, AssignReleasesTheOldTarget) {
  auto a = makeIntrusiveRefCnt<Thing>(1);
  auto b = makeIntrusiveRefCnt<Thing>(2);
  EXPECT_EQ(g_live.load(), 2);

  a = b;
  EXPECT_EQ(g_live.load(), 1);  // a's original target is gone
  EXPECT_EQ(a.Get(), b.Get());
  EXPECT_EQ(a.UseCount(), 2u);
}

// Copy-and-swap assignment makes these safe; a naive release-then-retain
// implementation frees the object out from under itself.
TEST_F(IntrusiveRefCntPtrTest, SelfAssignmentIsSafe) {
  auto a = makeIntrusiveRefCnt<Thing>(5);
  const IntrusiveRefCntPtr<Thing>& alias = a;
  a = alias;
  EXPECT_TRUE(a);
  EXPECT_EQ(a->value, 5);
  EXPECT_EQ(a.UseCount(), 1u);
  EXPECT_EQ(g_live.load(), 1);
}

TEST_F(IntrusiveRefCntPtrTest, SelfMoveIsSafe) {
  auto a = makeIntrusiveRefCnt<Thing>(6);
  IntrusiveRefCntPtr<Thing>& alias = a;
  a = std::move(alias);
  EXPECT_TRUE(a);
  EXPECT_EQ(a->value, 6);
  EXPECT_EQ(g_live.load(), 1);
}

TEST_F(IntrusiveRefCntPtrTest, AliasedAssignmentIsSafe) {
  auto a = makeIntrusiveRefCnt<Thing>(9);
  auto b = a;
  EXPECT_EQ(a.UseCount(), 2u);

  a = b;
  EXPECT_TRUE(a);
  EXPECT_EQ(a->value, 9);
  EXPECT_EQ(a.UseCount(), 2u);
  EXPECT_EQ(g_live.load(), 1);
}

TEST_F(IntrusiveRefCntPtrTest, Reset) {
  auto a = makeIntrusiveRefCnt<Thing>();
  a.Reset();
  EXPECT_FALSE(a);
  EXPECT_EQ(g_live.load(), 0);
}

TEST_F(IntrusiveRefCntPtrTest, Swap) {
  auto a = makeIntrusiveRefCnt<Thing>(1);
  auto b = makeIntrusiveRefCnt<Thing>(2);
  a.Swap(b);
  EXPECT_EQ(a->value, 2);
  EXPECT_EQ(b->value, 1);
  EXPECT_EQ(a.UseCount(), 1u);
}

// resetWithoutRelease drops the handle while keeping the reference alive. Since
// the raw-pointer constructor retains rather than adopts, reclaiming requires an
// explicit Release() to balance -- a sharp edge worth pinning.
TEST_F(IntrusiveRefCntPtrTest, ResetWithoutReleaseKeepsTheReference) {
  auto a = makeIntrusiveRefCnt<Thing>(4);
  Thing* raw = a.Get();
  a.ResetWithoutRelease();
  EXPECT_FALSE(a);
  EXPECT_EQ(g_live.load(), 1);  // still alive: the reference was not dropped

  IntrusiveRefCntPtr<Thing> reclaimed(raw);  // retains -> 2
  raw->Release();                            // drop the orphaned reference -> 1
  EXPECT_EQ(reclaimed.UseCount(), 1u);
  EXPECT_EQ(reclaimed->value, 4);
}

TEST_F(IntrusiveRefCntPtrTest, Comparison) {
  auto a = makeIntrusiveRefCnt<Thing>();
  auto b = makeIntrusiveRefCnt<Thing>();
  auto a2 = a;
  EXPECT_EQ(a, a2);
  EXPECT_NE(a, b);
  EXPECT_TRUE(a == a.Get());
  EXPECT_TRUE(a != nullptr);
  EXPECT_TRUE(IntrusiveRefCntPtr<Thing>() == nullptr);
}

TEST_F(IntrusiveRefCntPtrTest, ConstructFromUniquePtr) {
  auto p = IntrusiveRefCntPtr<Thing>(std::make_unique<Thing>(13));
  EXPECT_EQ(p->value, 13);
  EXPECT_EQ(p.UseCount(), 1u);
}

// ---------------------------------------------------------------------------
// Thread safety
//
// RefCountedBase and ThreadSafeRefCountedBase are equally easy to type, and
// choosing the former for a type that crosses threads is a silent data race
// with no symptom until production. These assertions make that choice explicit.
// ---------------------------------------------------------------------------

TEST(RefCountBase, EngineTypesUseTheAtomicBase) {
  static_assert(std::is_base_of_v<ThreadSafeRefCountedBase<Storage>, Storage>,
                "Storage crosses the scheduler/executor boundary");
  static_assert(
      std::is_base_of_v<ThreadSafeRefCountedBase<TensorImpl>, TensorImpl>,
      "TensorImpl crosses the scheduler/executor boundary");
  SUCCEED();
}

// The CRTP base deletes through Derived*, so no vtable is needed. Confirming it
// stays that way -- a stray virtual would add 8 bytes to every tensor.
TEST(RefCountBase, NoVtable) {
  static_assert(!std::is_polymorphic_v<Storage>);
  static_assert(!std::is_polymorphic_v<TensorImpl>);
  SUCCEED();
}

TEST_F(IntrusiveRefCntPtrTest, ConcurrentCopiesDoNotLoseCounts) {
  constexpr int kThreads = 8;
  constexpr int kIters = 20000;

  auto shared = makeIntrusiveRefCnt<Thing>(1);
  std::vector<std::thread> threads;
  threads.reserve(kThreads);

  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&shared] {
      for (int i = 0; i < kIters; ++i) {
        auto local = shared;  // retain
        EXPECT_EQ(local->value, 1);
        auto moved = std::move(local);
        EXPECT_TRUE(moved);
      }                       // release
    });
  }
  for (auto& th : threads) th.join();

  EXPECT_EQ(shared.UseCount(), 1u);
  EXPECT_EQ(g_live.load(), 1);
}

// The last handle to drop must destroy, exactly once, whichever thread wins.
TEST_F(IntrusiveRefCntPtrTest, ConcurrentReleaseDestroysExactlyOnce) {
  constexpr int kThreads = 16;
  for (int round = 0; round < 200; ++round) {
    auto owner = makeIntrusiveRefCnt<Thing>(round);
    std::vector<IntrusiveRefCntPtr<Thing>> handles(kThreads, owner);
    owner.Reset();

    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
      threads.emplace_back([&handles, t] { handles[t].Reset(); });
    }
    for (auto& th : threads) th.join();
    ASSERT_EQ(g_live.load(), 0) << "round " << round;
  }
}

}  // namespace
}  // namespace inferx
