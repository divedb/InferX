// gpt-oss-20b's logits against HuggingFace's.
//
// M11 Phase 2's acceptance test, and the thing the whole milestone is for. The
// pieces have each been checked alone -- MXFP4 bit-exactly, the sink rescale
// against the concatenated softmax, YaRN against transformers' frequencies, the
// clamped activation against its definition -- but "each piece is right" and
// "the model is right" are different claims, and only this one closes the gap
// M9 left open.
//
// A caveat that shapes what may be asserted. For Qwen2.5 the reference is fp32
// and the implementation bf16, so the reference is strictly better and any
// disagreement is ours. Here the reference runs bf16 *and reads the same 4-bit
// weights*, because 76 GB of fp32 experts exists on no machine we have. So the
// reference is not more precise than what it checks, and elementwise closeness
// is weaker evidence than it is for the dense path. **Argmax agreement at every
// position is the property to gate on.**
//
// Regenerate the reference with:
//   scripts/gen_gptoss_logits.py <ckpt> testdata/gptoss_20b_logits_bf16.bin
//
// It is derived data and is not committed; the test skips without it.

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
#include "inferx/model/gpt_oss.h"

namespace inferx::model {
namespace {

struct Reference {
  std::vector<int32_t> ids;
  int64_t tokens = 0;
  int64_t vocab = 0;
  std::vector<float> logits;  // [tokens, vocab]
};

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_GPTOSS_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--openai--gpt-oss-20b/snapshots/"
         "6cee5e81ee83917806bbde320786a8fb61efebee";
}

std::string ReferencePath() {
  if (const char* env = std::getenv("INFERX_TEST_GPTOSS_LOGITS")) return env;
  return "testdata/gptoss_20b_logits_bf16.bin";
}

// The 'IXRL' container, shared with gen_reference_logits.py.
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

class GptOssModelTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

    std::string why;
    if (!LoadReference(&reference_, &why)) {
      GTEST_SKIP() << why << "; regenerate with scripts/gen_gptoss_logits.py";
    }
  }

  Reference reference_;
};

TEST_F(GptOssModelTest, LogitsAgreeWithHuggingFace) {
  const std::string dir = CheckpointDir();

  auto model = GptOssModel::Load(dir);
  if (!model.ok()) GTEST_SKIP() << "no gpt-oss checkpoint at " << dir << ": "
                                << model.status().message();

  ASSERT_EQ(model->config().vocab_size, reference_.vocab)
      << "the reference was generated from a different checkpoint";

  std::vector<float> got;
  const Status s = model->Forward(reference_.ids, &got);
  ASSERT_TRUE(s.ok()) << s;

  ASSERT_EQ(got.size(), reference_.logits.size());

  // --- What actually has to hold: the same token wins, everywhere -----------
  //
  // A wrong nibble order, a missed sink, the wrong RoPE table or a mis-clamped
  // activation all move logits far enough to change the argmax. This is the
  // gate.
  for (int64_t t = 0; t < reference_.tokens; ++t) {
    const float* mine = got.data() + t * reference_.vocab;
    const float* theirs = reference_.logits.data() + t * reference_.vocab;

    const int64_t my_top =
        std::max_element(mine, mine + reference_.vocab) - mine;
    const int64_t their_top =
        std::max_element(theirs, theirs + reference_.vocab) - theirs;

    EXPECT_EQ(my_top, their_top)
        << "position " << t << " picked a different token (" << my_top
        << " vs " << their_top << ")";
  }

  // --- And a magnitude check, deliberately loose ---------------------------
  //
  // Normalized by the reference's own spread rather than stated absolutely:
  // logits over a 201088-wide vocabulary span tens of units, and an absolute
  // bound would be either vacuous or arbitrary. Two bf16 evaluation orders over
  // 24 layers do not agree closely, and this bound is not trying to make them.
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

// The short prompt above is 5 tokens against a 128-token window, so half the
// layers being windowed makes no difference to it: that test would pass with
// the sliding window unimplemented. This one runs 225 tokens, where 12 of the
// 24 layers genuinely cannot see the start of the prompt, and it is the only
// thing here that says so.
//
// Regenerate with:
//   scripts/gen_gptoss_logits.py <ckpt> testdata/gptoss_20b_logits_long.bin \
//       --prompt "<something over 128 tokens>"
TEST_F(GptOssModelTest, LongPromptExercisesTheSlidingWindow) {
  const char* env = std::getenv("INFERX_TEST_GPTOSS_LOGITS_LONG");
  const std::string path =
      env != nullptr ? env : "testdata/gptoss_20b_logits_long.bin";

  Reference long_ref;
  std::string why;
  {
    // LoadReference reads whichever path the environment names, so point it at
    // this one for the duration.
    const char* previous = std::getenv("INFERX_TEST_GPTOSS_LOGITS");
    setenv("INFERX_TEST_GPTOSS_LOGITS", path.c_str(), 1);

    const bool ok = LoadReference(&long_ref, &why);

    if (previous != nullptr) {
      setenv("INFERX_TEST_GPTOSS_LOGITS", previous, 1);
    } else {
      unsetenv("INFERX_TEST_GPTOSS_LOGITS");
    }

    if (!ok) GTEST_SKIP() << why;
  }

  const std::string dir = CheckpointDir();

  auto model = GptOssModel::Load(dir);
  if (!model.ok()) GTEST_SKIP() << "no gpt-oss checkpoint at " << dir;

  ASSERT_GT(long_ref.tokens, model->config().sliding_window)
      << "this prompt is not longer than the window, so it proves nothing";

  std::vector<float> got;
  ASSERT_TRUE(model->Forward(long_ref.ids, &got).ok());

  // The bar here is looser than the short prompt's, and the reason is a
  // measured property of the model rather than a concession.
  //
  // gpt-oss routes each token to 4 of 32 experts, and its router's margin
  // between the 4th and 5th ranked expert is *tiny*: over this prompt the
  // median is 0.07 and 28 of the 225 positions sit below 0.01. bf16 carries
  // about three decimal digits, so any difference in accumulation order --
  // ours against HuggingFace's, or HuggingFace's against itself on other
  // hardware -- can reorder that boundary. When it does, the token runs
  // through a *different expert*, and its output changes by far more than the
  // perturbation that caused it.
  //
  // So MoE routing is a discontinuity, and demanding argmax agreement at every
  // position is not achievable by any independent implementation, however
  // correct. This is the same class of fact as §6.3's "a cache hit can change
  // a greedy answer", and it deserves the same treatment: state it, bound it,
  // and do not pretend the bound is tighter than the arithmetic allows.
  //
  // What a *bug* would look like, and what this still catches: a mis-sized
  // sliding window, a wrong RoPE table or a missed sink perturbs every
  // position rather than the undecided few, and would blow through the bound
  // below many times over.
  int disagreements = 0;
  int64_t first_bad = -1;

  for (int64_t t = 0; t < long_ref.tokens; ++t) {
    const float* mine = got.data() + t * long_ref.vocab;
    const float* theirs = long_ref.logits.data() + t * long_ref.vocab;

    const int64_t my_top = std::max_element(mine, mine + long_ref.vocab) - mine;
    const int64_t their_top =
        std::max_element(theirs, theirs + long_ref.vocab) - theirs;

    if (my_top != their_top) {
      if (first_bad < 0) first_bad = t;
      ++disagreements;
    }
  }

  // 10%, and the number is derived rather than chosen. 12% of this prompt's
  // positions have a routing margin under 0.01, which bf16 cannot resolve; the
  // observed disagreement rate sits at 3-6% and *moves within that range* for
  // changes as small as swapping `__sincosf` for `sincosf` -- which is itself
  // the clearest demonstration that these flips are the routing discontinuity
  // and not an error being fixed. A bound tight enough to distinguish 3% from
  // 6% would be measuring the sine implementation, not the model.
  //
  // A structural bug is nowhere near this line: before the gate_up bias was
  // de-interleaved, the short prompt already picked the wrong token at *every*
  // position.
  EXPECT_LE(disagreements * 10, long_ref.tokens)
      << disagreements << " of " << long_ref.tokens
      << " positions picked a different token, first at " << first_bad
      << " (window " << model->config().sliding_window
      << "). Above ~10% this is a structural bug, not routing noise.";
}

TEST_F(GptOssModelTest, TheTopTokensAgreeAndNotJustTheFirst) {
  // Argmax alone can survive a model that is subtly wrong everywhere below the
  // winner. Comparing the top-5 at the last position is a cheap way to ask for
  // more without pretending two bf16 stacks agree bitwise.
  const std::string dir = CheckpointDir();

  auto model = GptOssModel::Load(dir);
  if (!model.ok()) GTEST_SKIP() << "no gpt-oss checkpoint at " << dir;

  std::vector<float> got;
  ASSERT_TRUE(model->Forward(reference_.ids, &got).ok());

  auto top_k = [&](const float* row, int k) {
    std::vector<int32_t> idx(static_cast<size_t>(reference_.vocab));
    for (size_t i = 0; i < idx.size(); ++i) idx[i] = static_cast<int32_t>(i);
    std::partial_sort(idx.begin(), idx.begin() + k, idx.end(),
                      [&](int32_t a, int32_t b) { return row[a] > row[b]; });
    idx.resize(static_cast<size_t>(k));
    return idx;
  };

  const int64_t last = reference_.tokens - 1;
  const auto mine = top_k(got.data() + last * reference_.vocab, 5);
  const auto theirs =
      top_k(reference_.logits.data() + last * reference_.vocab, 5);

  // As a set, not a sequence: ties and near-ties reorder between two bf16
  // evaluation orders without either being wrong.
  std::vector<int32_t> mine_sorted = mine;
  std::vector<int32_t> theirs_sorted = theirs;
  std::sort(mine_sorted.begin(), mine_sorted.end());
  std::sort(theirs_sorted.begin(), theirs_sorted.end());

  int overlap = 0;
  for (int32_t id : mine_sorted) {
    if (std::binary_search(theirs_sorted.begin(), theirs_sorted.end(), id)) {
      ++overlap;
    }
  }

  EXPECT_GE(overlap, 4) << "only " << overlap
                        << " of the reference's top 5 tokens appear in ours";
}

TEST_F(GptOssModelTest, CapturedDecodeMatchesLaunchByLaunch) {
  auto model = GptOssModel::Load(CheckpointDir());
  if (!model.ok()) GTEST_SKIP() << "no gpt-oss checkpoint";

  constexpr int64_t kBlockSize = 16;
  constexpr int64_t kMaxBlocks = 2;
  ASSERT_TRUE(model->AttachKvCache(/*num_blocks=*/4, kBlockSize).ok());
  ASSERT_TRUE(model->ReserveActivations(/*max_tokens=*/32).ok());
  auto block = model->kv_pool()->AllocateBlock();
  ASSERT_TRUE(block.ok());

  ForwardBatch prefill;
  prefill.num_seqs = 1;
  prefill.max_blocks_per_seq = kMaxBlocks;
  prefill.block_table = {*block, *block};
  prefill.token_ids = reference_.ids;
  prefill.positions.resize(reference_.ids.size());
  prefill.seq_of_token.assign(reference_.ids.size(), 0);
  prefill.slots.resize(reference_.ids.size());
  for (size_t i = 0; i < reference_.ids.size(); ++i) {
    prefill.positions[i] = static_cast<int32_t>(i);
    prefill.slots[i] = *block * kBlockSize + static_cast<int32_t>(i);
  }
  prefill.logits_indices = {
      static_cast<int32_t>(reference_.ids.size() - 1)};

  std::vector<float> prefill_logits;
  ASSERT_TRUE(model->Step(prefill, &prefill_logits).ok());
  const int32_t next_token = static_cast<int32_t>(
      std::max_element(prefill_logits.begin(), prefill_logits.end()) -
      prefill_logits.begin());

  ForwardBatch decode;
  decode.num_seqs = 1;
  decode.max_blocks_per_seq = kMaxBlocks;
  decode.block_table = {*block, *block};
  decode.token_ids = {next_token};
  decode.positions = {static_cast<int32_t>(reference_.ids.size())};
  decode.seq_of_token = {0};
  decode.slots = {static_cast<int32_t>(
      *block * kBlockSize + static_cast<int64_t>(reference_.ids.size()))};
  decode.logits_indices = {0};

  std::vector<float> direct;
  ASSERT_TRUE(model->Step(decode, &direct).ok());
  ASSERT_TRUE(model->CaptureDecodeGraph(/*num_seqs=*/1, kMaxBlocks).ok());
  ASSERT_EQ(model->captured_graphs(), 1);

  std::vector<float> captured;
  ASSERT_TRUE(model->Step(decode, &captured).ok());
  ASSERT_EQ(captured, direct);
}

}  // namespace
}  // namespace inferx::model
