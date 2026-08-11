/// End-to-end single-sequence prefill through FlashInfer's paged kernel.
///
/// Everything measured so far -- the M3-to-M4 progression, the bandwidth floor,
/// FP8 -- is a *decode* step at batch 1. Prefill was never benchmarked, and it
/// does not run the same code: decode uses FlashInfer's decode kernel, while a
/// prompt uses its separately planned ragged-prefill kernel. The reference
/// kernel remains the conformance oracle but is no longer the measured path.
///
/// Two questions, then, and this measures both:
///
///   1. What does a prompt cost, as a function of its length? Decode is
///      weight-bandwidth bound and flat in context; prefill is compute bound
///      and quadratic in it, so the two have nothing in common and the decode
///      numbers say nothing about time-to-first-token.
///   2. Does the tiled kernel keep scaling through the 16k context that the
///      reference kernel could not launch at all?
///
/// The arithmetic intensity line is the point of comparison. A prefill of N
/// tokens does the same weight reads as a decode step but N times the work, so
/// past a few hundred tokens it should leave the bandwidth floor behind
/// entirely and become a matter of how good the GEMMs are.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench/cuda_timer.h"
#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/model/qwen2.h"

namespace inferx {
namespace {

// Chosen to bracket real prompts and then keep going through long context:
// a chat turn is a few hundred tokens, a document is a few thousand, and the
// the former reference-kernel shared-memory ceiling was below 16k.
constexpr int64_t kLengths[] = {128, 256, 512, 1024, 2048, 4096, 8192, 16384};

constexpr int64_t kBlockSize = 16;

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

/// One sequence of `length` tokens, positions 0..length-1, laid out in
/// consecutive blocks. Only the last position's logits are wanted, which is
/// what a real prefill asks for.
model::ForwardBatch MakePrefillBatch(int64_t length,
                                     int64_t max_blocks_per_seq) {
  model::ForwardBatch b;
  b.num_seqs = 1;
  b.max_blocks_per_seq = max_blocks_per_seq;
  b.block_table.assign(static_cast<size_t>(max_blocks_per_seq), 0);

  const int64_t blocks = (length + kBlockSize - 1) / kBlockSize;

  for (int64_t k = 0; k < blocks; ++k) {
    b.block_table[static_cast<size_t>(k)] = static_cast<int32_t>(k);
  }

  b.token_ids.reserve(static_cast<size_t>(length));
  b.positions.reserve(static_cast<size_t>(length));
  b.seq_of_token.reserve(static_cast<size_t>(length));
  b.slots.reserve(static_cast<size_t>(length));

  for (int64_t i = 0; i < length; ++i) {
    // A fixed token id: prefill cost is a function of shape, not of content.
    b.token_ids.push_back(785);
    b.positions.push_back(static_cast<int32_t>(i));
    b.seq_of_token.push_back(0);
    b.slots.push_back(static_cast<int32_t>(
        b.block_table[static_cast<size_t>(i / kBlockSize)] * kBlockSize +
        i % kBlockSize));
  }

  b.logits_indices.push_back(static_cast<int32_t>(length - 1));
  return b;
}

}  // namespace
}  // namespace inferx

int main(int argc, char** argv) {
  using namespace inferx;         // NOLINT(build/namespaces)
  using namespace inferx::bench;  // NOLINT(build/namespaces) -- TimeLaunch

  int warmup = 3;
  int iters = 10;
  bool fp8 = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) {
      warmup = std::atoi(argv[++i]);
    } else if (arg == "--iters" && i + 1 < argc) {
      iters = std::atoi(argv[++i]);
    } else if (arg == "--fp8") {
      fp8 = true;
    } else {
      std::fprintf(stderr, "usage: %s [--warmup N] [--iters N] [--fp8]\n",
                   argv[0]);
      return 2;
    }
  }

  if (!cuda::Available()) {
    std::fprintf(stderr, "no CUDA device available\n");
    return 1;
  }

  auto loaded =
      model::Qwen2Model::LoadFromDirectory(CheckpointDir(), DeviceId::Cuda(0));
  if (!loaded.ok()) {
    std::fprintf(stderr, "cannot load model: %s\n",
                 loaded.status().ToString().c_str());
    return 1;
  }

  model::Qwen2Model model = *std::move(loaded);

  if (fp8) {
    if (const Status s = model.QuantizeWeightsToF8(); !s.ok()) {
      std::fprintf(stderr, "cannot quantize: %s\n", s.ToString().c_str());
      return 1;
    }
  }

  // The pool has to cover the longest prompt measured, and the block table has
  // to be wide enough for it.
  constexpr int64_t kLongest =
      kLengths[sizeof(kLengths) / sizeof(*kLengths) - 1];
  const int64_t max_blocks_per_seq = (kLongest + kBlockSize - 1) / kBlockSize;

  if (const Status s = model.AttachKvCache(max_blocks_per_seq + 8, kBlockSize);
      !s.ok()) {
    std::fprintf(stderr, "cannot attach KV cache: %s\n", s.ToString().c_str());
    return 1;
  }

  const auto& config = model.config();

  std::printf("%s\n", config.ToString().c_str());
  std::printf("weights: %s, %.2f GB\n", fp8 ? "fp8 e4m3" : "bf16",
              model.WeightBytes() / 1e9);
  std::printf("prefill, batch 1, one sequence\n\n");

  std::printf("%8s %12s %12s %12s %10s  %s\n", "tokens", "prefill_ms",
              "tok_per_s", "ms_per_tok", "noise_%", "note");
  std::printf("%s\n", std::string(78, '-').c_str());

  std::vector<float> logits;

  for (const int64_t length : kLengths) {
    // Sized to this prompt, as a scheduler's active block table would be.
    const int64_t blocks_for_this = (length + kBlockSize - 1) / kBlockSize;

    model::ForwardBatch b = MakePrefillBatch(length, blocks_for_this);

    auto timing =
        TimeLaunch([&] { return model.Step(b, &logits); }, warmup, iters);

    if (!timing.ok()) {
      // Keep failures in the table so a future context-length ceiling is a
      // benchmark result rather than a lost stack trace.
      std::printf("%8ld %12s %12s %12s %10s  %s\n", static_cast<long>(length),
                  "-", "-", "-", "-", timing.status().message().data());
      continue;
    }

    const double ms = timing->min_ms;

    std::printf("%8ld %12.3f %12.1f %12.4f %10.1f  %s\n",
                static_cast<long>(length), ms,
                static_cast<double>(length) / (ms * 1e-3), ms / length,
                timing->noise() * 100.0, length <= 512 ? "" : "");
  }

  // For contrast: one decode step at the same context, which is the same weight
  // traffic and one token of work. The ratio is what says whether prefill is
  // paying for compute or for weights.
  std::printf(
      "\nfor contrast, a single decode step is ~%.1f ms at this "
      "weight size,\nso a prompt is worth roughly that many decode "
      "steps' bandwidth regardless\nof its length -- everything above "
      "that is compute.\n",
      model.WeightBytes() / 736e9 * 1e3);

  return 0;
}
