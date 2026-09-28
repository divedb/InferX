// Tensor-parallel worker topology, end to end at world size 1: spawn the
// real worker process, handshake over the IPC channels, run one generation
// request through the full scheduler/runner stack, and shut it down. At
// larger worlds the same machinery runs -- the missing piece is the NCCL
// collectives backend, not the process or IPC layer.
#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <string>
#include <vector>

#include "inferx/config/parallel_config.h"
#include "inferx/core/device.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/dist/comm.h"
#include "inferx/dist/worker_ipc.h"
#include "inferx/dist/worker_launch.h"
#include "inferx/tokenizer/id.h"

#ifndef INFERX_BIN
#error "compile with -DINFERX_BIN=<path to the inferx executable>"
#endif

namespace inferx {
namespace dist {
namespace {

TEST(SingleRankCommTest, CollectivesAreIdentity) {
  SingleRankComm comm(ParallelConfig{});
  EXPECT_EQ(comm.size(), 1);
  EXPECT_EQ(comm.rank(), 0);
  ASSERT_TRUE(comm.Barrier().ok());

  // A world of one: reduce and gather return their input unchanged.
  constexpr int64_t kRows = 2, kCols = 3;
  const std::vector<uint16_t> bits = {0x3f80, 0x4000, 0x4040, 0xc000, 0x0000, 0x3f80};
  auto partial = Tensor::FromBlob(const_cast<uint16_t*>(bits.data()), DataType::kBFloat16,
                                  Shape({kRows, kCols}), DeviceId::Cpu());
  ASSERT_TRUE(partial.ok());
  EXPECT_TRUE(comm.AllReduceSumBf16(*partial).ok());

  auto gathered = Tensor::Empty(DataType::kBFloat16, Shape({kRows, kCols}), DeviceId::Cpu());
  ASSERT_TRUE(gathered.ok());
  ASSERT_TRUE(comm.AllGatherLastDim(*partial, *gathered).ok());
  const auto* out = static_cast<const uint16_t*>(gathered->Data());
  for (size_t i = 0; i < bits.size(); ++i) {
    EXPECT_EQ(out[i], bits[i]) << "element " << i;
  }
}

/// \brief One real worker process at rank 0 of world 1. The worker loads
///        the default model and engine configuration, exactly as spawned by
///        a controller.
class WorkerProcessTest : public ::testing::Test {
 protected:
  void SetUp() override {
    channels_ = MakeIpcChannelNames(++counter_);
    auto commands = MessageChannel::Create(channels_.commands);
    ASSERT_TRUE(commands.ok()) << commands.status();
    auto events = MessageChannel::Create(channels_.events);
    ASSERT_TRUE(events.ok()) << events.status();
    commands_ = std::move(*commands);
    events_ = std::move(*events);

    const std::vector<WorkerSpec> specs = {{/*rank=*/0, /*world=*/1, /*device=*/0}};
    auto pool = WorkerPool::Launch(specs, INFERX_BIN, /*passthrough=*/{}, channels_);
    ASSERT_TRUE(pool.ok()) << pool.status();
    pool_ = std::move(*pool);
  }

  void TearDown() override {
    if (pool_.has_value()) {
      WireMessage shutdown{};
      shutdown.kind = MessageKind::kShutdown;
      (void)commands_.Send(shutdown);
      (void)pool_->WaitAll();
    }
    commands_.Remove();
    events_.Remove();
  }

  bool ReceiveEvent(uint32_t timeout_ms, WireMessage* out) {
    return events_.TryReceive(timeout_ms, out);
  }

  MessageChannel commands_;
  MessageChannel events_;
  IpcChannelNames channels_;
  std::optional<WorkerPool> pool_;

 private:
  static std::atomic<uint64_t> counter_;
};

std::atomic<uint64_t> WorkerProcessTest::counter_{::getpid()};

TEST_F(WorkerProcessTest, GeneratesTokensAcrossProcessBoundary) {
  // Handshake: the worker announces its rank and world once loaded.
  WireMessage message{};
  ASSERT_TRUE(ReceiveEvent(30000, &message)) << "no handshake within 30s";
  ASSERT_EQ(message.kind, MessageKind::kReady);
  EXPECT_EQ(message.ready.rank, 0);
  EXPECT_EQ(message.ready.world, 1);

  // One short generation. Any valid token ids work as a prompt.
  WireMessage generate{};
  generate.kind = MessageKind::kGenerate;
  generate.generate.request_id = 7;
  generate.generate.num_prompt = 3;
  generate.generate.max_tokens = 4;
  generate.generate.prompt[0] = 9;
  generate.generate.prompt[1] = 42;
  generate.generate.prompt[2] = 1337;
  ASSERT_TRUE(commands_.Send(generate).ok());

  int tokens = 0;
  bool finished = false;
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (std::chrono::steady_clock::now() < deadline) {
    WireMessage event{};
    if (!ReceiveEvent(1000, &event)) continue;
    if (event.kind == MessageKind::kToken) {
      EXPECT_EQ(event.token.request_id, 7u);
      ++tokens;
    } else if (event.kind == MessageKind::kFinished) {
      EXPECT_EQ(event.finished.request_id, 7u);
      finished = true;
      break;
    } else if (event.kind == MessageKind::kFailed) {
      FAIL() << "worker failed: " << event.failed.text;
    }
  }
  EXPECT_GE(tokens, 1);
  EXPECT_TRUE(finished);

  // The explicit shutdown in TearDown completes the exit cleanly.
  WireMessage shutdown{};
  shutdown.kind = MessageKind::kShutdown;
  ASSERT_TRUE(commands_.Send(shutdown).ok());
  EXPECT_TRUE(pool_->WaitAll().ok());
  pool_.reset();
}

}  // namespace
}  // namespace inferx::dist
}  // namespace inferx
