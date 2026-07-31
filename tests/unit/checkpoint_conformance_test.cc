// Conformance against a real HuggingFace checkpoint.
//
// Skipped entirely when no checkpoint is present, so this stays runnable on a
// machine that has not downloaded one. Point INFERX_TEST_CHECKPOINT at a
// directory to override the default search.
//
// The synthetic tests in safetensors_test.cc cover the format's edge cases.
// This one covers the thing they cannot: that our reading of the format agrees
// with what a real publisher actually writes.

#include <cstdlib>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/tensor.h"
#include "inferx/model/safetensors.h"

namespace inferx::model {
namespace {

std::string FindCheckpoint() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  // The HF cache layout. The snapshot directory is a content hash, so this
  // hardcodes the one known to be present rather than globbing.
  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

class CheckpointConformance : public ::testing::Test {
 protected:
  void SetUp() override {
    dir_ = FindCheckpoint();

    auto opened = Checkpoint::Open(dir_);
    if (!opened.ok()) {
      GTEST_SKIP() << "no checkpoint at " << dir_ << " (" << opened.status()
                   << ")";
    }

    ckpt_ = std::make_unique<Checkpoint>(*std::move(opened));
  }

  std::string dir_;
  std::unique_ptr<Checkpoint> ckpt_;
};

// Qwen2.5-3B-Instruct: 36 layers, hidden 2048, 16 Q heads / 2 KV heads at
// head_dim 128, intermediate 11008, vocab 151936, tied embeddings.
constexpr int64_t kLayers = 36;
constexpr int64_t kHidden = 2048;
constexpr int64_t kHeadDim = 128;
constexpr int64_t kKvHeads = 2;
constexpr int64_t kIntermediate = 11008;
constexpr int64_t kVocab = 151936;

TEST_F(CheckpointConformance, OpensBothShardsAsOneNamespace) {
  EXPECT_EQ(ckpt_->shard_paths().size(), 2u);

  // 36 layers x 12 tensors + embed + final norm.
  EXPECT_EQ(ckpt_->size(), 434u);

  // Matches the index's declared total_size.
  EXPECT_EQ(ckpt_->TotalBytes(), 6171877376u);
}

TEST_F(CheckpointConformance, EveryTensorTheModelNeedsIsPresentAndShaped) {
  EXPECT_TRUE(ckpt_->GetChecked("model.embed_tokens.weight",
                                Shape({kVocab, kHidden}))
                  .ok());
  EXPECT_TRUE(ckpt_->GetChecked("model.norm.weight", Shape({kHidden})).ok());

  for (int64_t i = 0; i < kLayers; ++i) {
    const std::string p = "model.layers." + std::to_string(i) + ".";

    // GQA: Q is [hidden, hidden] but K and V are [kv_heads*head_dim, hidden].
    EXPECT_TRUE(ckpt_->GetChecked(p + "self_attn.q_proj.weight",
                                  Shape({kHidden, kHidden})).ok()) << p;
    EXPECT_TRUE(ckpt_->GetChecked(p + "self_attn.k_proj.weight",
                                  Shape({kKvHeads * kHeadDim, kHidden})).ok())
        << p;
    EXPECT_TRUE(ckpt_->GetChecked(p + "self_attn.v_proj.weight",
                                  Shape({kKvHeads * kHeadDim, kHidden})).ok())
        << p;
    EXPECT_TRUE(ckpt_->GetChecked(p + "self_attn.o_proj.weight",
                                  Shape({kHidden, kHidden})).ok()) << p;

    // Qwen2 puts a bias on Q/K/V but not on O -- a detail that silently breaks
    // the forward pass if assumed away.
    EXPECT_TRUE(ckpt_->GetChecked(p + "self_attn.q_proj.bias",
                                  Shape({kHidden})).ok()) << p;
    EXPECT_TRUE(ckpt_->GetChecked(p + "self_attn.k_proj.bias",
                                  Shape({kKvHeads * kHeadDim})).ok()) << p;
    EXPECT_TRUE(ckpt_->GetChecked(p + "self_attn.v_proj.bias",
                                  Shape({kKvHeads * kHeadDim})).ok()) << p;
    EXPECT_FALSE(ckpt_->Contains(p + "self_attn.o_proj.bias")) << p;

    EXPECT_TRUE(ckpt_->GetChecked(p + "mlp.gate_proj.weight",
                                  Shape({kIntermediate, kHidden})).ok()) << p;
    EXPECT_TRUE(ckpt_->GetChecked(p + "mlp.up_proj.weight",
                                  Shape({kIntermediate, kHidden})).ok()) << p;
    EXPECT_TRUE(ckpt_->GetChecked(p + "mlp.down_proj.weight",
                                  Shape({kHidden, kIntermediate})).ok()) << p;

    EXPECT_TRUE(ckpt_->GetChecked(p + "input_layernorm.weight",
                                  Shape({kHidden})).ok()) << p;
    EXPECT_TRUE(ckpt_->GetChecked(p + "post_attention_layernorm.weight",
                                  Shape({kHidden})).ok()) << p;
  }
}

// tie_word_embeddings is true, so there is no lm_head in the checkpoint and the
// output projection reuses the embedding matrix. Asserted rather than assumed:
// a loader that expects lm_head.weight fails at load on this model, and one
// that silently allocates a zero LM head produces uniform logits.
TEST_F(CheckpointConformance, HasNoLmHeadBecauseEmbeddingsAreTied) {
  EXPECT_FALSE(ckpt_->Contains("lm_head.weight"));
  EXPECT_TRUE(ckpt_->Contains("model.embed_tokens.weight"));
}

TEST_F(CheckpointConformance, WeightsAreBf16AndBorrowed) {
  auto embed = ckpt_->Get("model.embed_tokens.weight");
  ASSERT_TRUE(embed.ok()) << embed.status();

  EXPECT_EQ(embed->dtype(), DataType::kBFloat16);
  EXPECT_TRUE(embed->storage()->IsBorrowed());
  EXPECT_EQ(embed->nbytes(), kVocab * kHidden * 2);
}

// The load path must not copy: reading 6 GB of weights through a mapping should
// allocate essentially nothing of its own. Checked by touching every tensor and
// confirming each one still points inside a shard's mapping.
TEST_F(CheckpointConformance, NoTensorCopiesTheMapping) {
  int64_t total = 0;

  for (const std::string& name : ckpt_->Names()) {
    auto t = ckpt_->Get(name);
    ASSERT_TRUE(t.ok()) << name << ": " << t.status();
    ASSERT_TRUE(t->storage()->IsBorrowed()) << name;
    total += t->nbytes();
  }

  EXPECT_EQ(static_cast<size_t>(total), ckpt_->TotalBytes());
}

// A spot check that the bytes are real rather than an unfaulted mapping: BF16
// weights of a trained model are small, non-zero, and not all identical.
TEST_F(CheckpointConformance, WeightsLookLikeTrainedValues) {
  auto w = ckpt_->Get("model.layers.0.self_attn.q_proj.weight");
  ASSERT_TRUE(w.ok()) << w.status();

  const auto* bits = static_cast<const uint16_t*>(w->data());
  const int64_t n = w->numel();
  ASSERT_GT(n, 1024);

  int nonzero = 0;
  int distinct_prefix = 0;
  uint16_t first = bits[0];

  for (int64_t i = 0; i < 1024; ++i) {
    if (bits[i] != 0) ++nonzero;
    if (bits[i] != first) ++distinct_prefix;
  }

  EXPECT_GT(nonzero, 900) << "weights are mostly zero -- mapping not faulted?";
  EXPECT_GT(distinct_prefix, 900) << "weights are near-constant";
}

}  // namespace
}  // namespace inferx::model
