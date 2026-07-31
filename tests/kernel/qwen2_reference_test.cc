// M2's acceptance test: our logits against HuggingFace's.
//
// The reference is produced by scripts/gen_reference_logits.py running HF's own
// Qwen2ForCausalLM in fp32 on CPU. It shares nothing with this implementation
// except the checkpoint on disk, so agreement means our layer order, RoPE
// convention, GQA mapping, norm placement and tied-embedding handling match the
// architecture's definition rather than matching one person's reading of it.
//
// Two references are used, and the second is what makes this test rigorous.
//
// fp32 is ground truth. But our stack runs bf16, so "how close to fp32 is close
// enough" is a real question, and picking a number for it would be inventing a
// threshold rather than deriving one. So HF is also run in bf16, and *its*
// deviation from *its own* fp32 is the error budget bf16 costs. Our deviation
// then has to be no worse than that -- a bound calibrated by measurement rather
// than by taste, and one that tightens automatically if the reference improves.
//
// Regenerate both with:
//   scripts/gen_reference_logits.py <ckpt> testdata/qwen25_3b_logits_f32.bin
//   scripts/gen_reference_logits.py <ckpt> testdata/qwen25_3b_logits_bf16.bin \
//       --dtype bfloat16
//
// They are derived data and are not committed; the test skips without them.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <string>
#include <vector>

#include <gtest/gtest.h>

#include "inferx/core/cuda_utils.h"
#include "inferx/model/qwen2.h"

namespace inferx::model {
namespace {

struct Reference {
  std::vector<int32_t> ids;
  std::vector<float> logits;  // [tokens, vocab] row-major
  int64_t tokens = 0;
  int64_t vocab = 0;
};

bool LoadReference(const std::string& path, Reference* out) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) return false;

  char magic[4];
  uint32_t version = 0, tokens = 0, vocab = 0;

  const bool header_ok = std::fread(magic, 1, 4, f) == 4 &&
                         std::memcmp(magic, "IXRL", 4) == 0 &&
                         std::fread(&version, 4, 1, f) == 1 &&
                         std::fread(&tokens, 4, 1, f) == 1 &&
                         std::fread(&vocab, 4, 1, f) == 1 && version == 1;

  if (!header_ok) {
    std::fclose(f);
    return false;
  }

  out->tokens = tokens;
  out->vocab = vocab;
  out->ids.resize(tokens);
  out->logits.resize(static_cast<size_t>(tokens) * vocab);

  const bool body_ok =
      std::fread(out->ids.data(), sizeof(int32_t), tokens, f) == tokens &&
      std::fread(out->logits.data(), sizeof(float), out->logits.size(), f) ==
          out->logits.size();

  std::fclose(f);
  return body_ok;
}

std::string ReferenceDir() {
  if (const char* env = std::getenv("INFERX_REFERENCE_DIR")) return env;
  return "testdata";
}

/// Agreement statistics between two logit rows.
struct Agreement {
  double correlation = 0;
  double max_abs_diff = 0;
  double max_diff_in_sd = 0;
  double mean_shift = 0;
  double total_variation = 0;
};

std::vector<double> Softmax(const float* p, int64_t n) {
  const float max = *std::max_element(p, p + n);

  std::vector<double> out(static_cast<size_t>(n));
  double sum = 0;

  for (int64_t i = 0; i < n; ++i) {
    out[static_cast<size_t>(i)] = std::exp(p[i] - max);
    sum += out[static_cast<size_t>(i)];
  }
  for (double& x : out) x /= sum;

  return out;
}

/// Compares `a` against the fp32 truth `b`.
Agreement Compare(const float* a, const float* b, int64_t n) {
  double sum_a = 0, sum_b = 0;
  Agreement g;

  for (int64_t i = 0; i < n; ++i) {
    sum_a += a[i];
    sum_b += b[i];
    g.max_abs_diff = std::max<double>(g.max_abs_diff, std::abs(a[i] - b[i]));
  }

  const double mean_a = sum_a / n;
  const double mean_b = sum_b / n;

  double cov = 0, var_a = 0, var_b = 0;
  for (int64_t i = 0; i < n; ++i) {
    const double da = a[i] - mean_a;
    const double db = b[i] - mean_b;
    cov += da * db;
    var_a += da * da;
    var_b += db * db;
  }

  g.correlation = cov / std::sqrt(var_a * var_b);
  g.mean_shift = mean_a - mean_b;
  g.max_diff_in_sd = g.max_abs_diff / std::sqrt(var_b / n);

  const std::vector<double> pa = Softmax(a, n);
  const std::vector<double> pb = Softmax(b, n);

  double tv = 0;
  for (int64_t i = 0; i < n; ++i) {
    tv += std::abs(pa[static_cast<size_t>(i)] - pb[static_cast<size_t>(i)]);
  }
  g.total_variation = 0.5 * tv;

  return g;
}

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

class ReferenceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (!CudaAvailable()) GTEST_SKIP() << "no CUDA device available";

    const std::string dir = ReferenceDir();

    if (!LoadReference(dir + "/qwen25_3b_logits_f32.bin", &ref_) ||
        !LoadReference(dir + "/qwen25_3b_logits_bf16.bin", &ref_bf16_)) {
      GTEST_SKIP() << "no references in " << dir
                   << "; regenerate with scripts/gen_reference_logits.py";
    }

    ASSERT_EQ(ref_.ids, ref_bf16_.ids) << "the two references used different "
                                          "prompts";

    auto loaded = Qwen2Model::LoadFromDirectory(CheckpointDir());
    if (!loaded.ok()) GTEST_SKIP() << "model not loaded: " << loaded.status();

    model_ = std::make_unique<Qwen2Model>(*std::move(loaded));

    ASSERT_TRUE(model_->Forward(ref_.ids, &ours_).ok());
    ASSERT_EQ(ours_.size(), ref_.logits.size());
  }

  const float* Ours(int64_t t) const { return ours_.data() + t * ref_.vocab; }
  const float* Fp32(int64_t t) const {
    return ref_.logits.data() + t * ref_.vocab;
  }
  const float* HfBf16(int64_t t) const {
    return ref_bf16_.logits.data() + t * ref_.vocab;
  }

  Reference ref_;       // HF fp32 -- ground truth
  Reference ref_bf16_;  // HF bf16 -- the error budget bf16 costs
  std::vector<float> ours_;
  std::unique_ptr<Qwen2Model> model_;
};

// The headline: the predicted token must match at every position, not just the
// last. A model that is right at the end and wrong in the middle has a bug that
// happens not to bite on this prompt.
TEST_F(ReferenceTest, ArgmaxMatchesAtEveryPosition) {
  for (int64_t t = 0; t < ref_.tokens; ++t) {
    const auto ours = std::distance(
        Ours(t), std::max_element(Ours(t), Ours(t) + ref_.vocab));
    const auto theirs = std::distance(
        Fp32(t), std::max_element(Fp32(t), Fp32(t) + ref_.vocab));

    EXPECT_EQ(ours, theirs)
        << "position " << t << ": we predict " << ours << " (logit "
        << Ours(t)[ours] << "), HF predicts " << theirs << " (logit "
        << Fp32(t)[theirs] << ")";
  }
}

// Ranking agreement below top-1, over the prefix of the ranking that bf16 can
// actually resolve.
//
// Naively comparing the top 5 fails, and measuring why is instructive: at
// position 4 the rank-5 and rank-6 logits are 17.2559 and 17.2468, a gap of
// 0.0091, while bf16's own error at that position is 0.16-0.23. Those two
// tokens are separated by six percent of the noise floor -- their order is a
// coin flip in any bf16 implementation, including HF's own. Demanding we
// reproduce it would be demanding we reproduce fp32 rounding, not correctness.
//
// So the comparison runs down the fp32 ranking and stops at the first pair the
// error budget cannot separate. Everything above that line is a real ordering
// and must match exactly.
TEST_F(ReferenceTest, RankingAgreesWhereverBf16CanResolveIt) {
  for (int64_t t = 0; t < ref_.tokens; ++t) {
    // The budget is per pair, not global. Whether ranks r and r+1 can be told
    // apart depends on how much bf16 moved *those two logits*, not on the worst
    // single logit anywhere in a 151936-wide vocabulary -- using the global
    // maximum makes even the top token look unresolvable, which it plainly is
    // not, since the argmax test passes.
    const auto pair_budget = [&](int32_t a, int32_t b) {
      return std::abs(HfBf16(t)[a] - Fp32(t)[a]) +
             std::abs(HfBf16(t)[b] - Fp32(t)[b]);
    };

    constexpr int kMaxRank = 16;

    std::vector<int32_t> order(static_cast<size_t>(ref_.vocab));
    std::iota(order.begin(), order.end(), 0);
    std::partial_sort(order.begin(), order.begin() + kMaxRank + 1, order.end(),
                      [&](int32_t a, int32_t b) {
                        return Fp32(t)[a] > Fp32(t)[b];
                      });

    int resolvable = 0;
    while (resolvable < kMaxRank) {
      const int32_t hi = order[static_cast<size_t>(resolvable)];
      const int32_t lo = order[static_cast<size_t>(resolvable) + 1];

      if (Fp32(t)[hi] - Fp32(t)[lo] <= pair_budget(hi, lo)) break;
      ++resolvable;
    }

    std::printf("  position %ld: bf16 resolves the top %d of the ranking\n",
                static_cast<long>(t), resolvable);

    // Zero is a legitimate answer, not a broken one. Position 2 here predicts
    // after "The capital of", where the top two candidates sit closer together
    // than bf16's own error on them -- genuinely ambiguous text, not a
    // degenerate prompt. Such a position contributes nothing checkable to a
    // ranking comparison; it is still covered by the correlation and
    // total-variation bounds below.
    if (resolvable == 0) continue;

    std::vector<int32_t> ours(static_cast<size_t>(ref_.vocab));
    std::iota(ours.begin(), ours.end(), 0);
    std::partial_sort(ours.begin(), ours.begin() + resolvable, ours.end(),
                      [&](int32_t a, int32_t b) {
                        return Ours(t)[a] > Ours(t)[b];
                      });

    for (int r = 0; r < resolvable; ++r) {
      EXPECT_EQ(ours[static_cast<size_t>(r)], order[static_cast<size_t>(r)])
          << "position " << t << ", rank " << r;
    }
  }
}

// The acceptance criterion, and the reason both references exist: our distance
// from fp32 must be no worse than the distance bf16 costs HF itself. The factor
// of 1.5 is slack for a different reduction order and a different GEMM
// implementation, not for a different answer.
TEST_F(ReferenceTest, WeAreNoFurtherFromFp32ThanBf16Itself) {
  constexpr double kSlack = 1.5;

  for (int64_t t = 0; t < ref_.tokens; ++t) {
    const Agreement ours = Compare(Ours(t), Fp32(t), ref_.vocab);
    const Agreement hf = Compare(HfBf16(t), Fp32(t), ref_.vocab);

    std::printf(
        "  position %ld:  ours corr=%.6f maxd=%.3fsd tv=%.5f  |  "
        "hf-bf16 corr=%.6f maxd=%.3fsd tv=%.5f\n",
        static_cast<long>(t), ours.correlation, ours.max_diff_in_sd,
        ours.total_variation, hf.correlation, hf.max_diff_in_sd,
        hf.total_variation);

    EXPECT_LT(1.0 - ours.correlation, (1.0 - hf.correlation) * kSlack)
        << "position " << t << ": correlation shortfall exceeds bf16's own";

    EXPECT_LT(ours.max_diff_in_sd, hf.max_diff_in_sd * kSlack)
        << "position " << t << ": worst logit deviates more than bf16 explains";

    EXPECT_LT(ours.total_variation, hf.total_variation * kSlack)
        << "position " << t << ": sampling distribution differs more than "
                               "bf16 explains";

    // A systematic offset points at a wrong epsilon or a missing scale rather
    // than at rounding, so it is bounded absolutely rather than relatively.
    EXPECT_LT(std::abs(ours.mean_shift), 0.15) << "position " << t;
  }
}

}  // namespace
}  // namespace inferx::model
