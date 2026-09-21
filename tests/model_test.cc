#include "inferx/models/model.h"

#include <filesystem>
#include <fstream>
#include <string>

#include "gtest/gtest.h"
#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/device.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/shape.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"

namespace inferx {
namespace {

class ModelTest : public ::testing::Test {
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

  /// \brief A minimal valid ModelInput: one single-token sequence.
  StatusOr<ModelInput> MakeInput(int num_tokens) {
    const DeviceId device = DeviceId::Cuda(0);
    auto i32 = [&](Shape shape) { return Tensor::Empty(DataType::kInt32, shape, device); };
    INFERX_ASSIGN_OR_RETURN(Tensor token_ids, i32(Shape({num_tokens})));
    INFERX_ASSIGN_OR_RETURN(Tensor positions, i32(Shape({num_tokens})));
    INFERX_ASSIGN_OR_RETURN(Tensor batch_indices, i32(Shape({num_tokens})));
    INFERX_ASSIGN_OR_RETURN(Tensor qo_indptr, i32(Shape({2})));
    INFERX_ASSIGN_OR_RETURN(Tensor kv_indptr, i32(Shape({2})));
    INFERX_ASSIGN_OR_RETURN(Tensor kv_indices, i32(Shape({1})));
    INFERX_ASSIGN_OR_RETURN(Tensor last_page_len, i32(Shape({1})));
    INFERX_ASSIGN_OR_RETURN(Tensor logit_rows, i32(Shape({1})));
    std::vector<int32_t> pos(num_tokens);
    for (int i = 0; i < num_tokens; ++i) pos[i] = i;
    const std::vector<int32_t> zeros(num_tokens, 0);
    INFERX_RETURN_IF_ERROR(runtime_->Copy(
        token_ids.Data(), pos.data(), num_tokens * sizeof(int32_t), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(runtime_->Copy(
        positions.Data(), pos.data(), num_tokens * sizeof(int32_t), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(runtime_->Copy(batch_indices.Data(), zeros.data(),
                                          num_tokens * sizeof(int32_t),
                                          CopyKind::kHostToDevice));
    const int32_t qo[] = {0, num_tokens};
    const int32_t kv[] = {0, 1};
    const int32_t block[] = {0};
    const int32_t page[] = {num_tokens};
    const int32_t row[] = {num_tokens - 1};
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(qo_indptr.Data(), qo, sizeof(qo), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(kv_indptr.Data(), kv, sizeof(kv), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(kv_indices.Data(), block, sizeof(block), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(last_page_len.Data(), page, sizeof(page), CopyKind::kHostToDevice));
    INFERX_RETURN_IF_ERROR(
        runtime_->Copy(logit_rows.Data(), row, sizeof(row), CopyKind::kHostToDevice));

    ModelInput input;
    input.token_ids = token_ids;
    input.attention = {
        positions, batch_indices, qo_indptr, kv_indptr, kv_indices, last_page_len, {},
        {},        num_tokens,    1};
    input.logit_rows = logit_rows;
    return input;
  }

  DeviceRuntime* runtime_ = nullptr;
  Stream stream_;
};

TEST_F(ModelTest, LoadsQwen3CheckpointAndConfig) {
  auto model = Model::Load("models/Qwen3-0.6B", DeviceId::Cuda(0), /*max_tokens=*/8,
                           /*max_seqs=*/2);
  ASSERT_TRUE(model.ok()) << model.status();
  const ModelConfig& config = (*model)->config();
  EXPECT_EQ(config.model_type, "qwen3");
  EXPECT_EQ(config.architectures, "Qwen3ForCausalLM");
  EXPECT_EQ(config.hidden_size, 1024);
  EXPECT_EQ(config.intermediate_size, 3072);
  EXPECT_EQ(config.num_hidden_layers, 28);
  EXPECT_EQ(config.num_attention_heads, 16);
  EXPECT_EQ(config.num_key_value_heads, 8);
  EXPECT_EQ(config.head_dim, 128);
  EXPECT_EQ(config.vocab_size, 151936);
  EXPECT_FLOAT_EQ(config.rms_norm_eps, 1e-6f);
  EXPECT_FLOAT_EQ(config.rope_theta, 1e6f);
  EXPECT_TRUE(config.tie_word_embeddings);
}

TEST_F(ModelTest, ForwardReportsPendingOps) {
  auto model = Model::Load("models/Qwen3-0.6B", DeviceId::Cuda(0), /*max_tokens=*/8,
                           /*max_seqs=*/2);
  ASSERT_TRUE(model.ok()) << model.status();
  const ModelConfig& config = (*model)->config();
  KvLayout layout;
  layout.entries_per_token = 2;
  layout.kv_heads = config.num_key_value_heads;
  layout.head_dim = config.head_dim;
  layout.dtype = DataType::kBFloat16;
  auto pool = KvBlockPool::Create(config.num_hidden_layers, /*num_kv_blocks=*/4,
                                  /*block_size=*/2, layout, DeviceId::Cuda(0));
  ASSERT_TRUE(pool.ok());
  auto input = MakeInput(2);
  ASSERT_TRUE(input.ok());
  ops::ExecutionContext ctx(*runtime_, stream_);
  const StatusOr<Tensor> logits = (*model)->Forward(*input, *pool, ctx);
  EXPECT_EQ(logits.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(ModelTest, RejectsUnsupportedArchitecture) {
  const std::string dir = ::testing::TempDir() + "inferx-unsupported-arch";
  std::filesystem::create_directories(dir);
  std::ofstream(dir + "/config.json")
      << R"({"model_type": "gpt_neox", "architectures": ["GPTNeoXForCausalLM"],)"
      << R"( "hidden_size": 8, "num_hidden_layers": 1, "num_attention_heads": 1,)"
      << R"( "vocab_size": 16})";
  auto model = Model::Load(dir, DeviceId::Cuda(0), 8, 2);
  EXPECT_EQ(model.status().code(), absl::StatusCode::kUnimplemented);
}

TEST_F(ModelTest, RejectsMissingConfig) {
  auto model = Model::Load("/nonexistent/model-dir", DeviceId::Cuda(0), 8, 2);
  EXPECT_FALSE(model.ok());
}

}  // namespace
}  // namespace inferx
