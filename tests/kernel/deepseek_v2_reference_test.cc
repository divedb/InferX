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
// The RoPE-convention question this file was written to settle is settled:
// de-interleaved won (2026-08-09, post-rope q/k matched HF to bf16 rounding)
// and is hardcoded in the loader. The other systematic trap this gate caught:
// transformers' integrated DeepseekV2 drops the yarn mscale^2 softmax-scale
// factor the checkpoint's own modeling code applies, so references must come
// from scripts/gen_deepseek_logits.py, which patches it back.
//
// The gpt-oss caveat applies here too: V2-Lite routes each token to 6 of 64
// experts, and a 6th/7th router margin inside bf16 noise can route an
// independent implementation differently, moving that token's logits by more
// than the perturbation. Short prompts rarely hit it; if a single position
// disagrees, check the router margin before concluding the model is wrong.

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "inferx/core/cuda_utils.h"
#include "inferx/model/deepseek_v2.h"
#include "inferx/scheduler/scheduler.h"

namespace inferx::model {
namespace {

struct Reference {
  std::vector<int32_t> ids;
  int64_t tokens = 0;
  int64_t vocab = 0;
  std::vector<float> logits;  // [tokens, vocab]
};

struct DecodeReference {
  std::vector<int32_t> prompt;
  std::vector<int32_t> generated;
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
      std::fread(&version, 4, 1, f) != 1 || std::fread(&tokens, 4, 1, f) != 1 ||
      std::fread(&vocab, 4, 1, f) != 1 || version != 1) {
    *why = path + ": bad header";
    std::fclose(f);
    return false;
  }

  out->tokens = tokens;
  out->vocab = vocab;
  out->ids.resize(tokens);
  out->logits.resize(static_cast<size_t>(tokens) * vocab);

  const bool ok = std::fread(out->ids.data(), 4, tokens, f) == tokens &&
                  std::fread(out->logits.data(), 4, out->logits.size(), f) ==
                      out->logits.size();

  std::fclose(f);

  if (!ok) {
    *why = path + ": truncated body";
    return false;
  }

  return true;
}

std::string DecodeReferencePath() {
  if (const char* env = std::getenv("INFERX_TEST_DEEPSEEK_DECODE")) return env;
  return "testdata/deepseek_v2_lite_decode.bin";
}

bool LoadDecodeReference(DecodeReference* out, std::string* why) {
  const std::string path = DecodeReferencePath();
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (f == nullptr) {
    *why = "no cached-decode reference at " + path;
    return false;
  }

  char magic[4];
  uint32_t version = 0;
  uint32_t prompt = 0;
  uint32_t generated = 0;
  constexpr uint32_t kMaxReferenceTokens = 1U << 20;
  if (std::fread(magic, 1, 4, f) != 4 || std::memcmp(magic, "IXDG", 4) != 0 ||
      std::fread(&version, 4, 1, f) != 1 || std::fread(&prompt, 4, 1, f) != 1 ||
      std::fread(&generated, 4, 1, f) != 1 || version != 1 || prompt == 0 ||
      generated == 0 || prompt > kMaxReferenceTokens ||
      generated > kMaxReferenceTokens) {
    *why = path + ": bad header";
    std::fclose(f);
    return false;
  }

  out->prompt.resize(prompt);
  out->generated.resize(generated);
  const bool ok =
      std::fread(out->prompt.data(), sizeof(int32_t), prompt, f) == prompt &&
      std::fread(out->generated.data(), sizeof(int32_t), generated, f) ==
          generated;
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
      GTEST_SKIP() << why << "; regenerate with scripts/gen_deepseek_logits.py";
    }
  }

  Reference reference_;
};

TEST_F(DeepseekV2ReferenceTest, LogitsAgreeWithHuggingFace) {
  const std::string dir = CheckpointDir();

  auto model = DeepseekV2Model::Load(dir, DeviceId::Cuda(0));
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
  //
  // One escape hatch: positions where the reference's own top-2 margin is
  // within a couple of bf16 ULPs (the chat prompt has *exact* ties at three
  // of 65 positions) don't identify a winner at bf16 at all, so a
  // disagreement there is tie-breaking, not error. kTieTolerance is two ULPs
  // at the logit magnitudes this model produces (~32).
  constexpr float kTieTolerance = 0.25f;
  int tie_flips = 0;
  for (int64_t t = 0; t < reference_.tokens; ++t) {
    const float* mine = got.data() + t * reference_.vocab;
    const float* theirs = reference_.logits.data() + t * reference_.vocab;

    const int64_t my_top =
        std::max_element(mine, mine + reference_.vocab) - mine;
    const int64_t their_top =
        std::max_element(theirs, theirs + reference_.vocab) - theirs;

    if (my_top != their_top &&
        theirs[their_top] - theirs[my_top] <= kTieTolerance) {
      ++tie_flips;
      continue;
    }

    EXPECT_EQ(my_top, their_top)
        << "position " << t << " picked a different token (" << my_top << " vs "
        << their_top << "), reference margin "
        << theirs[their_top] - theirs[my_top]
        << "; if MANY positions disagree by wide margins, check the softmax "
           "scale (yarn mscale^2) and that the reference was generated by "
           "scripts/gen_deepseek_logits.py, which restores it over "
           "transformers' integrated port";
  }
  if (tie_flips > 0) {
    std::fprintf(stderr,
                 "note: %d position(s) disagreed inside the reference's own "
                 "bf16 tie tolerance\n",
                 tie_flips);
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

TEST_F(DeepseekV2ReferenceTest,
       CachedGreedyDecodeMatchesHuggingFaceAtEveryStep) {
  DecodeReference reference;
  std::string why;
  if (!LoadDecodeReference(&reference, &why)) {
    GTEST_SKIP() << why
                 << "; regenerate with scripts/gen_deepseek_logits.py "
                    "--decode-output <path> --max-new-tokens <n>";
  }

  auto model = DeepseekV2Model::Load(CheckpointDir(), DeviceId::Cuda(0));
  if (!model.ok()) {
    GTEST_SKIP() << "no DeepSeek-V2 checkpoint: " << model.status().message();
  }

  // Compute InferX's own full-prefix logits over the HF continuation before
  // attaching the serving cache. At a failure this separates a cached-decode
  // defect (cached token != full-prefix token) from a model-wide numerical or
  // routing difference (both InferX paths agree but differ from HF).
  std::vector<int32_t> teacher_forced = reference.prompt;
  teacher_forced.insert(teacher_forced.end(), reference.generated.begin(),
                        reference.generated.end());
  std::vector<float> full_logits;
  ASSERT_TRUE(model->Forward(teacher_forced, &full_logits).ok());

  const int64_t total_tokens = static_cast<int64_t>(reference.prompt.size() +
                                                    reference.generated.size());

  // The margin oracle: the IXRL container must span the same teacher-forced
  // sequence (gen_deepseek_logits.py writes it that way when asked for a
  // decode reference), so a cached-path argmax flip can be classified by
  // HF's own top-2 margin at that position rather than failing outright.
  ASSERT_EQ(reference_.tokens, total_tokens)
      << "the logits reference does not span prompt+generated; regenerate "
         "both artifacts together with --decode-output";
  for (int64_t t = 0; t < total_tokens; ++t) {
    ASSERT_EQ(reference_.ids[static_cast<size_t>(t)],
              teacher_forced[static_cast<size_t>(t)])
        << "logits and decode references disagree at token " << t
        << "; they were generated from different runs";
  }
  constexpr int64_t kBlockSize = 16;
  const int64_t blocks = (total_tokens + kBlockSize - 1) / kBlockSize;
  ASSERT_TRUE(model->AttachKvCache(blocks + 1, kBlockSize).ok());

  scheduler::SchedulerConfig config;
  config.max_running = 1;
  config.max_batch_tokens =
      std::max<int64_t>(1, static_cast<int64_t>(reference.prompt.size()));
  config.max_seq_len = total_tokens + 1;
  config.enable_prefix_cache = false;
  auto scheduler = scheduler::Scheduler::Create(config, model->kv_pool());
  ASSERT_TRUE(scheduler.ok()) << scheduler.status();

  scheduler::SamplingParams params;
  params.max_tokens = static_cast<int32_t>(reference.generated.size());
  params.temperature = 0.0f;
  ASSERT_TRUE(scheduler->AddRequest(/*id=*/1, reference.prompt, params).ok());

  size_t generated = 0;
  while (generated < reference.generated.size()) {
    ForwardBatch batch;
    ASSERT_TRUE(scheduler->PrepareStep(&batch).ok())
        << "preparing decode step " << generated;
    ASSERT_FALSE(batch.token_ids.empty())
        << "scheduler stopped before reference token " << generated;

    std::vector<float> logits;
    ASSERT_TRUE(model->Step(batch, &logits).ok()) << "model step " << generated;

    std::vector<int32_t> sampled;
    const int64_t vocab = model->config().vocab_size;
    ASSERT_EQ(logits.size(),
              batch.logits_indices.size() * static_cast<size_t>(vocab));
    sampled.reserve(batch.logits_indices.size());
    for (size_t row = 0; row < batch.logits_indices.size(); ++row) {
      const float* begin = logits.data() + row * static_cast<size_t>(vocab);
      const int32_t token =
          static_cast<int32_t>(std::max_element(begin, begin + vocab) - begin);
      ASSERT_LT(generated, reference.generated.size());

      const size_t full_row = reference.prompt.size() + generated - 1;
      const float* full_begin =
          full_logits.data() + full_row * static_cast<size_t>(vocab);
      const int32_t full_token = static_cast<int32_t>(
          std::max_element(full_begin, full_begin + vocab) - full_begin);

      const int32_t ref_token = reference.generated[generated];

      // A flip inside HF's own top-2 margin (a couple of bf16 ULPs) is
      // tie-breaking between two candidates the reference itself cannot
      // separate, not a cache defect. Teacher-forcing the reference token
      // below keeps the trajectory aligned either way, so one tolerated tie
      // does not invalidate every later step.
      constexpr float kTieTolerance = 0.25f;
      if (token != ref_token) {
        const float* ref_row =
            reference_.logits.data() + full_row * static_cast<size_t>(vocab);
        const float ref_margin = ref_row[ref_token] - ref_row[token];
        const float cached_margin = begin[token] - begin[ref_token];

        ASSERT_LE(ref_margin, kTieTolerance)
            << "first cached greedy divergence at generated token " << generated
            << " (position " << reference.prompt.size() + generated
            << "): cached picked " << token << " over the reference's "
            << ref_token << " (cached margin " << cached_margin
            << "), and HF's own margin " << ref_margin
            << " is too wide to be a bf16 tie. InferX full-prefix argmax="
            << full_token;
        std::fprintf(stderr,
                     "note: generated token %zu flipped inside the "
                     "reference's tie tolerance (ref margin %.4f, cached "
                     "margin %.4f)\n",
                     generated, ref_margin, cached_margin);
      }

      sampled.push_back(ref_token);
      ++generated;
    }

    ASSERT_TRUE(scheduler->CommitStep(sampled).ok())
        << "committing decode step " << generated;
  }

  const auto completed = scheduler->TakeCompleted();
  ASSERT_EQ(completed.size(), 1);
  EXPECT_EQ(completed[0].output_tokens, reference.generated);
}

}  // namespace
}  // namespace inferx::model
