// DeepSeek-V2-Lite's logits against HuggingFace's.
//
// The §18.7 D5 acceptance gate, and the only thing that can settle what the
// self-consistent tests cannot: whether our half-split RoPE convention matches
// the checkpoint's layout (ARCHITECTURE.md's standing caveat), whether the
// YaRN table and mscale softmax scale reproduce HF's numbers, and whether the
// dense/MoE schedule, routing and ungated shared experts compose into the
// same model HF runs.
//
// Both inputs are derived data and are not committed; the test skips without
// them. Generate on a machine that holds the checkpoint (§18.6 chose a rented
// bf16 GPU; docs/DSV2_VALIDATION.md is the runbook):
//
//   scripts/gen_deepseek_logits.py <ckpt> testdata/deepseek_v2_lite_logits.bin
//
// If the argmax gate fails, rerun the suite with
// INFERX_DSV2_ROPE_DEINTERLEAVE=1 -- whichever convention passes gets
// hardcoded and the toggle removed. That decision procedure is the reason
// this file exists before any machine can run it.
//
// The gpt-oss caveat applies here too: V2-Lite routes each token to 6 of 64
// experts, and a 6th/7th router margin inside bf16 noise can route an
// independent implementation differently, moving that token's logits by more
// than the perturbation. Short prompts rarely hit it; if a single position
// disagrees, check the router margin before concluding the model is wrong.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/model/deepseek_v2.h"

namespace inferx::model {
namespace {

struct Reference {
  std::vector<int32_t> ids;
  int64_t tokens = 0;
  int64_t vocab = 0;
  std::vector<float> logits;  // [tokens, vocab]
};

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_DEEPSEEK_CHECKPOINT")) {
    return env;
  }

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  // The HF cache layout `huggingface-cli download` produces; the snapshot
  // hash varies, so the runbook sets the env var instead of relying on this.
  return std::string(home) +
         "/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V2-Lite/"
         "snapshots/main";
}

std::string ReferencePath() {
  if (const char* env = std::getenv("INFERX_TEST_DEEPSEEK_LOGITS")) return env;
  return "testdata/deepseek_v2_lite_logits.bin";
}

// The 'IXRL' container, shared with scripts/gen_deepseek_logits.py.
bool LoadReference(Reference* out, std::string* why) {
  const std::string path = ReferencePath();

  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    *why = "no reference logits at " + path;
    return false;
  }

  char magic[4];
  uint32_t version = 0;
  uint32_t tokens = 0;
  uint32_t vocab = 0;

  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "IXRL", 4) != 0 ||
      std::fread(&version, 4, 1, f) != 1 ||
      std::fread(&tokens, 4, 1, f) != 1 ||
      std::fread(&vocab, 4, 1, f) != 1 || version != 1) {
    *why = path + ": bad header";
    std::fclose(f);
    return false;
  }

  out->tokens = tokens;
  out->vocab = vocab;
  out->ids.resize(tokens);
  out->logits.resize(static_cast<size_t>(tokens) * vocab);

  const bool ok =
      std::fread(out->ids.data(), 4, tokens, f) == tokens &&
      std::fread(out->logits.data(), 4, out->logits.size(), f) ==
          out->logits.size();

  std::fclose(f);

  if (!ok) {
    *why = path + ": truncated body";
    return false;
  }

  return true;
}

class DeepseekV2ReferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

    std::string why;
    if (!LoadReference(&reference_, &why)) {
      GTEST_SKIP() << why
                   << "; regenerate with scripts/gen_deepseek_logits.py";
    }
  }

  Reference reference_;
};

TEST_F(DeepseekV2ReferenceTest, LogitsAgreeWithHuggingFace) {
  const std::string dir = CheckpointDir();

  auto model = DeepseekV2Model::Load(dir);
  if (!model.ok()) {
    GTEST_SKIP() << "no DeepSeek-V2-Lite checkpoint at " << dir << ": "
                 << model.status().message();
  }

  ASSERT_EQ(model->config().vocab_size, reference_.vocab)
      << "the reference was generated from a different checkpoint";

  std::vector<float> got;
  const Status s = model->Forward(reference_.ids, &got);
  ASSERT_TRUE(s.ok()) << s;

  ASSERT_EQ(got.size(), reference_.logits.size());

  // --- The gate: the same token wins at every position ----------------------
  //
  // A wrong RoPE interleave, a missing mscale, a swapped gate/up in the
  // stacked experts, or a dense layer run as MoE all move logits far enough
  // to change the argmax somewhere in even a short prompt.
  for (int64_t t = 0; t < reference_.tokens; ++t) {
    const float* mine = got.data() + t * reference_.vocab;
    const float* theirs = reference_.logits.data() + t * reference_.vocab;

    const int64_t my_top =
        std::max_element(mine, mine + reference_.vocab) - mine;
    const int64_t their_top =
        std::max_element(theirs, theirs + reference_.vocab) - theirs;

    EXPECT_EQ(my_top, their_top)
        << "position " << t << " picked a different token (" << my_top
        << " vs " << their_top
        << "); if every position disagrees, try the RoPE convention toggle "
           "(INFERX_DSV2_ROPE_DEINTERLEAVE=1)";
  }

  // --- And a magnitude check, deliberately loose ---------------------------
  //
  // Normalized by the reference's own spread. Against a bf16 GPU reference two
  // evaluation orders over 27 layers do not agree closely; against an --fp32
  // reference this bound could be tightened, and the runbook says to prefer
  // one when the rented box's RAM allows it.
  double ref_span = 0.0;
  for (int64_t t = 0; t < reference_.tokens; ++t) {
    const float* theirs = reference_.logits.data() + t * reference_.vocab;
    const auto [lo, hi] =
        std::minmax_element(theirs, theirs + reference_.vocab);
    ref_span = std::max<double>(ref_span, *hi - *lo);
  }
  ASSERT_GT(ref_span, 0.0);

  double worst = 0.0;
  for (size_t i = 0; i < got.size(); ++i) {
    worst = std::max<double>(worst, std::abs(got[i] - reference_.logits[i]));
  }

  EXPECT_LT(worst / ref_span, 0.05)
      << "worst logit deviation " << worst << " against a reference spread of "
      << ref_span;
}

}  // namespace
}  // namespace inferx::model
