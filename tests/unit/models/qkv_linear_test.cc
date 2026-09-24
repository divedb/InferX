#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
#include <unistd.h>
#include <vector>

#include "gtest/gtest.h"
#include "inferx/core/device.h"
#include "inferx/core/tensor.h"
#include "inferx/engine/parallel_config.h"
#include "inferx/models/causal/decoder_stack.h"
#include "inferx/models/causal/weight_mapping.h"
#include "inferx/models/checkpoint.h"
#include "inferx/models/components/attention.h"
#include "inferx/models/components/qkv_linear.h"
#include "inferx/models/model_registry.h"

namespace inferx {
namespace {

components::AttentionConfig Attention(int64_t query_heads, int64_t kv_heads, int64_t head_dim) {
  components::AttentionConfig a;
  a.query_heads = query_heads;
  a.kv_heads = kv_heads;
  a.head_dim = head_dim;
  return a;
}

ParallelConfig Parallel(int size, int rank) { return ParallelConfig{size, rank}; }

float Bf16ToFloat(uint16_t bits) {
  const uint32_t wide = static_cast<uint32_t>(bits) << 16;
  float value = 0.0f;
  std::memcpy(&value, &wide, sizeof(value));
  return value;
}

TEST(ShardQkvTest, SingleRankIsIdentity) {
  const auto g = components::ShardQkv(Attention(4, 2, 64), Parallel(1, 0));
  ASSERT_TRUE(g.ok()) << g.status();
  EXPECT_EQ(g->query_heads, 4);
  EXPECT_EQ(g->kv_heads, 2);
  EXPECT_EQ(g->query_rows, 256);
  EXPECT_EQ(g->kv_rows, 128);
  EXPECT_EQ(g->kv_head_replicas, 1);
  EXPECT_EQ(g->kv_shard, 0);
}

TEST(ShardQkvTest, HeadsDivideAcrossRanks) {
  const auto g = components::ShardQkv(Attention(32, 8, 128), Parallel(4, 1));
  ASSERT_TRUE(g.ok()) << g.status();
  EXPECT_EQ(g->query_heads, 8);
  EXPECT_EQ(g->kv_heads, 2);
  EXPECT_EQ(g->query_rows, 8 * 128);
  EXPECT_EQ(g->kv_rows, 2 * 128);
  EXPECT_EQ(g->kv_head_replicas, 1);
  EXPECT_EQ(g->kv_shard, 1);
}

TEST(ShardQkvTest, KvHeadsReplicateWhenFewerThanRanks) {
  for (int rank = 0; rank < 8; ++rank) {
    const auto g = components::ShardQkv(Attention(32, 4, 64), Parallel(8, rank));
    ASSERT_TRUE(g.ok()) << g.status();
    EXPECT_EQ(g->kv_heads, 1);
    EXPECT_EQ(g->kv_head_replicas, 2);
    EXPECT_EQ(g->kv_shard, rank / 2) << "rank " << rank;
    EXPECT_EQ(g->query_heads, 4);
  }
}

TEST(ShardQkvTest, GatedOutputDoublesQueryRows) {
  auto total = Attention(8, 4, 64);
  total.output_gate = components::OutputGate::kSigmoid;
  const auto g = components::ShardQkv(total, Parallel(2, 1));
  ASSERT_TRUE(g.ok()) << g.status();
  EXPECT_EQ(g->query_heads, 4);
  EXPECT_EQ(g->query_rows, 4 * 64 * 2);
}

TEST(ShardQkvTest, RejectsBadTopologyAndIndivisibleHeads) {
  EXPECT_FALSE(components::ShardQkv(Attention(32, 8, 64), Parallel(0, 0)).ok());
  EXPECT_FALSE(components::ShardQkv(Attention(32, 8, 64), Parallel(4, 4)).ok());
  EXPECT_FALSE(components::ShardQkv(Attention(32, 8, 64), Parallel(4, -1)).ok());
  EXPECT_FALSE(components::ShardQkv(Attention(33, 33, 64), Parallel(4, 0)).ok());
  EXPECT_FALSE(components::ShardQkv(Attention(32, 6, 64), Parallel(4, 0)).ok());
  EXPECT_FALSE(components::ShardQkv(Attention(32, 3, 64), Parallel(4, 0)).ok());
}

/// \brief A tiny one-layer llama-layout checkpoint on disk, f32, whose q/k/v
///        rows encode their part and row index in their values.
class QkvLoadingTest : public ::testing::Test {
 protected:
  static constexpr int64_t kHidden = 8;
  static constexpr int64_t kQueryRows = 16;  // 4 heads * head_dim 4.
  static constexpr int64_t kKvRows = 8;      // 2 heads * head_dim 4.

  void SetUp() override {
    const std::string test_name =
        ::testing::UnitTest::GetInstance()->current_test_info()->name();
    dir_ = std::filesystem::temp_directory_path() /
           ("inferx_qkv_linear_" + std::to_string(::getpid()) + "_" + test_name);
    std::filesystem::create_directories(dir_);
  }

  void TearDown() override {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }

  /// \brief One named f32 tensor; rows encode `base + row index` per element.
  struct TensorSpec {
    std::string name;
    std::vector<int64_t> shape;
    float base;  // Row i is uniformly `base + i`; bases stay below 256 so
                 // every value is exact in bf16.
  };

  void WriteCheckpoint(const std::vector<TensorSpec>& tensors) {
    std::string header = "{";
    std::vector<std::byte> blob;
    bool first = true;
    for (const auto& t : tensors) {
      int64_t numel = 1;
      for (auto d : t.shape) numel *= d;
      const int64_t cols = t.shape.back();
      const int64_t begin = static_cast<int64_t>(blob.size());
      for (int64_t f = 0; f < numel; ++f) {
        // Element f is uniform within its row; row index f / cols encodes
        // which checkpoint row it came from.
        const float value = t.base + static_cast<float>(f / cols);
        std::byte raw[4];
        std::memcpy(raw, &value, sizeof(raw));
        for (auto b : raw) blob.push_back(b);
      }
      const int64_t end = static_cast<int64_t>(blob.size());
      if (!first) header += ",";
      first = false;
      header += "\"" + t.name + "\":{\"dtype\":\"F32\",\"shape\":[" +
                std::to_string(t.shape[0]);
      for (size_t d = 1; d < t.shape.size(); ++d) header += "," + std::to_string(t.shape[d]);
      header += "],\"data_offsets\":[" + std::to_string(begin) + "," + std::to_string(end) +
                "]}";
    }
    header += "}";

    std::vector<std::byte> file(8 + header.size() + blob.size(), std::byte{0});
    for (unsigned i = 0; i < 8; ++i) {
      file[i] = static_cast<std::byte>(header.size() >> (8 * i));
    }
    std::memcpy(file.data() + 8, header.data(), header.size());
    std::memcpy(file.data() + 8 + header.size(), blob.data(), blob.size());

    std::ofstream out(dir_ / "model.safetensors", std::ios::binary | std::ios::trunc);
    out.write(reinterpret_cast<const char*>(file.data()),
              static_cast<std::streamsize>(file.size()));
  }

  /// \brief The full tensor set of a one-layer llama-layout checkpoint.
  std::vector<TensorSpec> StandardTensors() {
    const std::string layer = "model.layers.0.";
    return {
        {"model.embed_tokens.weight", {8, kHidden}, 0.0f},
        {"model.norm.weight", {kHidden}, 0.0f},
        {layer + "input_layernorm.weight", {kHidden}, 0.0f},
        {layer + "post_attention_layernorm.weight", {kHidden}, 0.0f},
        {layer + "self_attn.q_proj.weight", {kQueryRows, kHidden}, 0.0f},
        {layer + "self_attn.k_proj.weight", {kKvRows, kHidden}, 100.0f},
        {layer + "self_attn.v_proj.weight", {kKvRows, kHidden}, 200.0f},
        {layer + "self_attn.o_proj.weight", {kHidden, kQueryRows}, 0.0f},
        {layer + "mlp.gate_proj.weight", {16, kHidden}, 0.0f},
        {layer + "mlp.up_proj.weight", {16, kHidden}, 0.0f},
        {layer + "mlp.down_proj.weight", {kHidden, 16}, 0.0f},
    };
  }

  StatusOr<causal::DecoderConfig> Config() {
    const std::string json =
        "{\"model_type\":\"llama\",\"hidden_size\":8,\"intermediate_size\":16,"
        "\"num_hidden_layers\":1,\"num_attention_heads\":4,\"num_key_value_heads\":2,"
        "\"head_dim\":4,\"vocab_size\":8,\"max_position_embeddings\":64,"
        "\"rms_norm_eps\":0.00001,\"rope_theta\":10000.0,\"tie_word_embeddings\":false}";
    CheckpointConfig identity;
    identity.model_type = "llama";
    return TranslateFamilyConfig(identity, json);
  }

  StatusOr<causal::DecoderWeights> Load(const std::vector<TensorSpec>& tensors,
                                        const ParallelConfig& parallel) {
    WriteCheckpoint(tensors);
    INFERX_ASSIGN_OR_RETURN(auto checkpoint, models::Checkpoint::Open(dir_.string()));
    INFERX_ASSIGN_OR_RETURN(auto config, Config());
    return causal::LoadDecoderWeights(checkpoint, config, causal::CheckpointLayout{},
                                      parallel, DeviceId::Cpu());
  }

  float RowValue(const Tensor& packed, int64_t row, int64_t width = kHidden) {
    const auto* bits = static_cast<const uint16_t*>(packed.Data());
    return Bf16ToFloat(bits[row * width]);
  }

  std::filesystem::path dir_;
};

TEST_F(QkvLoadingTest, PacksQkvInBlockContiguousOrder) {
  const auto weights = Load(StandardTensors(), Parallel(1, 0));
  ASSERT_TRUE(weights.ok()) << weights.status();
  const auto& attn = weights->blocks[0].mixer;
  EXPECT_EQ(attn.packed_qkv.GetShape().ToString(),
            Shape({kQueryRows + 2 * kKvRows, kHidden}).ToString());
  for (int64_t row = 0; row < kQueryRows; ++row) {
    EXPECT_FLOAT_EQ(RowValue(attn.packed_qkv, row), static_cast<float>(row));
  }
  for (int64_t row = 0; row < kKvRows; ++row) {
    EXPECT_FLOAT_EQ(RowValue(attn.packed_qkv, kQueryRows + row), 100.0f + row);
    EXPECT_FLOAT_EQ(RowValue(attn.packed_qkv, kQueryRows + kKvRows + row), 200.0f + row);
  }
}

TEST_F(QkvLoadingTest, PerProjectionWeightsAreViewsIntoThePackedTensor) {
  const auto weights = Load(StandardTensors(), Parallel(1, 0));
  ASSERT_TRUE(weights.ok()) << weights.status();
  const auto& attn = weights->blocks[0].mixer;
  const auto* base = static_cast<const std::byte*>(attn.packed_qkv.Data());
  EXPECT_EQ(static_cast<const std::byte*>(attn.query.weight.Data()), base);
  EXPECT_EQ(static_cast<const std::byte*>(attn.key.weight.Data()),
            base + kQueryRows * kHidden * 2);
  EXPECT_EQ(static_cast<const std::byte*>(attn.value.weight.Data()),
            base + (kQueryRows + kKvRows) * kHidden * 2);
}

TEST_F(QkvLoadingTest, ShardsQueryAndKvRowsByRank) {
  // 4 q heads / 2 kv heads at tp=2, rank 1: query rows 8..15, kv shard 1
  // (rows 4..7 of k_proj and v_proj).
  const auto weights = Load(StandardTensors(), Parallel(2, 1));
  ASSERT_TRUE(weights.ok()) << weights.status();
  const auto& attn = weights->blocks[0].mixer;
  ASSERT_EQ(attn.packed_qkv.Dim(0), 8 + 2 * 4);
  for (int64_t row = 0; row < 8; ++row) {
    EXPECT_FLOAT_EQ(RowValue(attn.packed_qkv, row), 8.0f + row);
  }
  for (int64_t row = 0; row < 4; ++row) {
    EXPECT_FLOAT_EQ(RowValue(attn.packed_qkv, 8 + row), 104.0f + row);
    EXPECT_FLOAT_EQ(RowValue(attn.packed_qkv, 8 + 4 + row), 204.0f + row);
  }
}

TEST_F(QkvLoadingTest, ReplicatesKvHeadsBelowRankCount) {
  // 2 kv heads at tp=4, rank 3: one replicated kv head, shard 3/2 = 1 ->
  // k/v rows 4..7; query rows 12..15 (rank * 1 q head * head_dim 4).
  const auto weights = Load(StandardTensors(), Parallel(4, 3));
  ASSERT_TRUE(weights.ok()) << weights.status();
  const auto& attn = weights->blocks[0].mixer;
  ASSERT_EQ(attn.packed_qkv.Dim(0), 4 + 2 * 4);  // 1 q head, 1 kv head.
  for (int64_t row = 0; row < 4; ++row) {
    EXPECT_FLOAT_EQ(RowValue(attn.packed_qkv, row), 12.0f + row);
    EXPECT_FLOAT_EQ(RowValue(attn.packed_qkv, 4 + row), 104.0f + row);
    EXPECT_FLOAT_EQ(RowValue(attn.packed_qkv, 4 + 4 + row), 204.0f + row);
  }
}

TEST_F(QkvLoadingTest, MissingProjectionIsNotFound) {
  auto tensors = StandardTensors();
  tensors.erase(tensors.begin() + 6);  // v_proj
  const auto weights = Load(tensors, Parallel(1, 0));
  ASSERT_FALSE(weights.ok());
  EXPECT_EQ(weights.status().code(), absl::StatusCode::kNotFound);
  EXPECT_NE(weights.status().message().find("v_proj.weight"), std::string::npos);
}

TEST_F(QkvLoadingTest, WrongProjectionShapeIsRejected) {
  auto tensors = StandardTensors();
  tensors[4].shape = {12, kHidden};  // q_proj
  const auto weights = Load(tensors, Parallel(1, 0));
  ASSERT_FALSE(weights.ok());
  EXPECT_EQ(weights.status().code(), absl::StatusCode::kInvalidArgument);
  EXPECT_NE(weights.status().message().find("q_proj.weight"), std::string::npos);
}

}  // namespace
}  // namespace inferx
