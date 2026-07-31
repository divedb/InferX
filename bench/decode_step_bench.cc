// Decode step latency: launch-by-launch against CUDA graph replay.
//
// A decode step through Qwen2.5-3B is roughly 400 launches -- 36 layers of
// seven GEMMs and half a dozen small kernels each, plus the head. At batch 1
// none of them is large, so the step can be bound by the cost of *issuing*
// work rather than doing it. bench/attention_bench.cc already showed the
// attention kernel alone behaving that way; this measures the whole step.
//
// That is the entire claim of §5.2 and M6: get the CPU off the critical path.
// A graph replaces the 400 launches with one replay, so what is measured here
// is how much of a decode step was launch overhead.
//
// The two paths do NOT run identical kernels, and that is the finding rather
// than a flaw in the measurement. FlashInfer plans on the host from the batch's
// sequence lengths, so it cannot go inside a graph -- a replay would reuse a
// plan built for whatever lengths were current at capture. Capture therefore
// records the reference kernel, which reads everything from device buffers and
// is length-agnostic. The two optimizations are exclusive today, and this is
// what says which one to keep.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "bench/cuda_timer.h"
#include "inferx/core/cuda_utils.h"
#include "inferx/model/qwen2.h"

namespace inferx::bench {
namespace {

std::string CheckpointDir() {
  if (const char* env = std::getenv("INFERX_TEST_CHECKPOINT")) return env;

  const char* home = std::getenv("HOME");
  if (home == nullptr) return "";

  return std::string(home) +
         "/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/"
         "aa8e72537993ba99e69dfaafa59ed015b17504d1";
}

constexpr int64_t kBatches[] = {1, 2, 4, 8, 16, 32};
constexpr int64_t kContext = 256;

/// Builds a decode batch: one token per sequence, each at position `context-1`.
model::ForwardBatch MakeDecodeBatch(int64_t batch, int64_t context,
                                    int64_t block_size,
                                    int64_t max_blocks_per_seq) {
  model::ForwardBatch b;
  b.num_seqs = batch;
  b.max_blocks_per_seq = max_blocks_per_seq;
  b.block_table.assign(static_cast<size_t>(batch * max_blocks_per_seq), 0);

  const int64_t blocks_per_seq = (context + block_size - 1) / block_size;

  for (int64_t s = 0; s < batch; ++s) {
    for (int64_t k = 0; k < blocks_per_seq; ++k) {
      b.block_table[static_cast<size_t>(s * max_blocks_per_seq + k)] =
          static_cast<int32_t>(s * blocks_per_seq + k);
    }

    const int64_t pos = context - 1;
    const int32_t block = static_cast<int32_t>(
        b.block_table[static_cast<size_t>(s * max_blocks_per_seq +
                                          pos / block_size)]);

    b.token_ids.push_back(785);
    b.positions.push_back(static_cast<int32_t>(pos));
    b.seq_of_token.push_back(static_cast<int32_t>(s));
    b.slots.push_back(
        static_cast<int32_t>(block * block_size + (pos % block_size)));
  }

  b.logits_indices.push_back(0);
  return b;
}

int Main(int argc, char** argv) {
  int warmup = 5;
  int iters = 30;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc) warmup = std::atoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc) iters = std::atoi(argv[++i]);
    else {
      std::fprintf(stderr, "usage: %s [--warmup N] [--iters N]\n", argv[0]);
      return 2;
    }
  }

  if (!CudaAvailable()) {
    std::fprintf(stderr, "no CUDA device available\n");
    return 1;
  }

  auto loaded = model::Qwen2Model::LoadFromDirectory(CheckpointDir());
  if (!loaded.ok()) {
    std::fprintf(stderr, "cannot load model: %s\n",
                 loaded.status().ToString().c_str());
    return 1;
  }

  model::Qwen2Model model = *std::move(loaded);

  const int64_t block_size = 16;
  const int64_t max_blocks_per_seq = 64;

  if (const Status s = model.AttachKvCache(2048, block_size); !s.ok()) {
    std::fprintf(stderr, "cannot attach KV cache: %s\n",
                 s.ToString().c_str());
    return 1;
  }

  std::printf("%s\n", model.config().ToString().c_str());
  std::printf("decode step, context %ld, one token per sequence\n\n",
              static_cast<long>(kContext));

  // The two columns are not two ways of issuing the same work. FlashInfer plans
  // on the host against the batch's sequence lengths, so it cannot be captured;
  // a graph therefore records the reference kernel instead. What is compared is
  // "FlashInfer, launch by launch" against "reference kernel, replayed", and
  // the header says so rather than calling the ratio a speedup.
  std::printf("%6s %14s %14s %9s  %s\n", "batch", "flashinfer_ms",
              "graph+ref_ms", "noise_%", "graph/fi");
  std::printf("%s\n", std::string(62, '-').c_str());

  int failures = 0;

  for (const int64_t batch : kBatches) {
    const model::ForwardBatch b =
        MakeDecodeBatch(batch, kContext, block_size, max_blocks_per_seq);

    std::vector<float> logits;

    // Launch-by-launch, before any graph exists for this shape.
    auto launches = TimeLaunch([&] { return model.Step(b, &logits); },
                               warmup, iters);
    if (!launches.ok()) {
      std::fprintf(stderr, "batch %ld failed: %s\n", static_cast<long>(batch),
                   launches.status().ToString().c_str());
      ++failures;
      continue;
    }

    if (const Status s = model.CaptureDecodeGraph(batch, max_blocks_per_seq);
        !s.ok()) {
      std::fprintf(stderr, "batch %ld capture failed: %s\n",
                   static_cast<long>(batch), s.ToString().c_str());
      ++failures;
      continue;
    }

    // The same call: Step now finds a graph for this shape and replays it.
    auto graphed = TimeLaunch([&] { return model.Step(b, &logits); }, warmup,
                              iters);
    if (!graphed.ok()) {
      std::fprintf(stderr, "batch %ld graphed failed: %s\n",
                   static_cast<long>(batch),
                   graphed.status().ToString().c_str());
      ++failures;
      continue;
    }

    std::printf("%6ld %14.4f %14.4f %9.1f  %6.2fx\n", static_cast<long>(batch),
                launches->min_ms, graphed->min_ms, graphed->noise() * 100.0,
                launches->min_ms / graphed->min_ms);
  }

  std::printf("\ncaptured graphs: %ld\n",
              static_cast<long>(model.captured_graphs()));

  return failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
