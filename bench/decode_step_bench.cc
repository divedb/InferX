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

#include <algorithm>
#include <chrono>
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
    const int32_t block =
        static_cast<int32_t>(b.block_table[static_cast<size_t>(
            s * max_blocks_per_seq + pos / block_size)]);

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
  bool fp8 = false;
  bool fp8_kv = false;

  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    if (arg == "--warmup" && i + 1 < argc)
      warmup = std::atoi(argv[++i]);
    else if (arg == "--iters" && i + 1 < argc)
      iters = std::atoi(argv[++i]);
    else if (arg == "--fp8")
      fp8 = true;
    else if (arg == "--fp8-kv")
      fp8_kv = true;
    else {
      std::fprintf(stderr,
                   "usage: %s [--warmup N] [--iters N] [--fp8] [--fp8-kv]\n",
                   argv[0]);
      return 2;
    }
  }

  if (!CudaAvailable()) {
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

  const int64_t block_size = 16;
  const int64_t max_blocks_per_seq = 64;

  // EnableFp8KvCache must precede AttachKvCache: the pool's element type is
  // fixed when it is allocated. Scales freeze later, in the first warmup Step
  // before any capture.
  if (fp8_kv) {
    if (const Status s = model.EnableFp8KvCache(); !s.ok()) {
      std::fprintf(stderr, "cannot enable fp8 KV: %s\n", s.ToString().c_str());
      return 1;
    }
  }

  if (const Status s = model.AttachKvCache(2048, block_size); !s.ok()) {
    std::fprintf(stderr, "cannot attach KV cache: %s\n", s.ToString().c_str());
    return 1;
  }

  // Before any capture: both quantization and sampling change what a graph
  // would record, and both refuse once one exists.
  // Sized for the largest batch measured, not a fixed 8. Sampling allocates
  // one output slot per logits row, so a smaller reservation made the last two
  // batches fail inside the timing loop -- reported as a capture error, which
  // reads like a graph bug and is not one.
  constexpr int64_t kMaxBatch =
      *std::max_element(std::begin(kBatches), std::end(kBatches));

  // Sized for the largest batch before anything is captured. FP8's activation
  // staging buffer is allocated on demand, so capturing batch 1 and then
  // preparing batch 32 reallocated it and left graph 1 replaying freed memory
  // -- an illegal access that the sustained loop below used to swallow and
  // report as 582,000 tokens a second.
  if (const Status s = model.ReserveActivations(kMaxBatch); !s.ok()) {
    std::fprintf(stderr, "reserve activations: %s\n", s.ToString().c_str());
    return 1;
  }

  if (const Status s = model.EnableDeviceSampling(kMaxBatch); !s.ok()) {
    std::fprintf(stderr, "device sampling: %s\n", s.ToString().c_str());
    return 1;
  }

  if (fp8) {
    const Status q = model.QuantizeWeightsToF8();
    if (!q.ok()) {
      std::fprintf(stderr, "cannot quantize: %s\n", q.ToString().c_str());
      return 1;
    }
  }

  std::printf("%s\n", model.config().ToString().c_str());
  std::printf("weights: %s, %.2f GB\n", fp8 ? "fp8 e4m3" : "bf16",
              model.WeightBytes() / 1e9);
  std::printf("kv cache: %s, %.2f GB (%.0fM tokens at this pool size)\n",
              fp8_kv ? "fp8 e4m3" : "bf16", model.kv_pool()->bytes() / 1e9,
              static_cast<double>(model.kv_pool()->num_blocks() *
                                  model.kv_pool()->block_size()) /
                  1e6);
  std::printf("decode step, context %ld, one token per sequence\n\n",
              static_cast<long>(kContext));

  // Both columns now run the same kernels, FlashInfer included: the planner is
  // hoisted out of the captured region, so a graph contains the fast attention
  // rather than the reference one. This is a like-for-like comparison of
  // dispatch again, which it was not before.
  std::printf("%6s %14s %14s %9s %9s %9s  %s\n", "batch", "launches_ms",
              "graph_ms", "device_ms", "host_ms", "noise_%", "speedup");
  std::printf("%s\n", std::string(82, '-').c_str());

  int failures = 0;

  for (const int64_t batch : kBatches) {
    const model::ForwardBatch b =
        MakeDecodeBatch(batch, kContext, block_size, max_blocks_per_seq);

    std::vector<float> logits;

    // Launch-by-launch, before any graph exists for this shape.
    auto launches =
        TimeLaunch([&] { return model.Step(b, &logits); }, warmup, iters);
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
    auto graphed =
        TimeLaunch([&] { return model.Step(b, &logits); }, warmup, iters);
    if (!graphed.ok()) {
      std::fprintf(stderr, "batch %ld graphed failed: %s\n",
                   static_cast<long>(batch),
                   graphed.status().ToString().c_str());
      ++failures;
      continue;
    }

    // The device portion, minimum over its own samples so it compares like for
    // like against graph_ms. Reading it once after the timing loop would put a
    // single sample against a minimum and produce a negative remainder, which
    // is how this was first written and how it was caught.
    double device_ms = 1e9;
    for (int i = 0; i < iters; ++i) {
      if (!model.Step(b, &logits).ok()) break;
      device_ms = std::min(device_ms, model.last_step_device_ms());
    }

    std::printf("%6ld %14.4f %14.4f %9.4f %9.4f %9.1f  %6.2fx\n",
                static_cast<long>(batch), launches->min_ms, graphed->min_ms,
                device_ms, graphed->min_ms - device_ms,
                graphed->noise() * 100.0, launches->min_ms / graphed->min_ms);
  }

  std::printf("\ncaptured graphs: %ld\n",
              static_cast<long>(model.captured_graphs()));

  // --- §5.2 depth 0 against depth 1 -----------------------------------------
  // Sustained generation rather than a single step, because that is where the
  // pipeline shows: overlap hides the host's per-step work behind the GPU, and
  // a one-shot measurement has nothing to hide it behind.
  {
    constexpr int64_t kBatch = 1;
    constexpr int kSteps = 60;

    model::ForwardBatch b =
        MakeDecodeBatch(kBatch, kContext, block_size, max_blocks_per_seq);

    std::vector<float> logits;
    std::vector<int32_t> tokens;

    // Checked, not discarded. A swallowed failure here does not look like a
    // failure -- the loop simply returns instantly and the benchmark reports
    // hundreds of thousands of tokens a second, which is how the FP8 path's
    // broken sustained loop went unnoticed.
    const auto must = [&](const Status& s, const char* what) {
      if (!s.ok()) {
        std::fprintf(stderr, "sustained %s failed: %s\n", what,
                     s.ToString().c_str());
        std::exit(1);
      }
    };

    // Depth 0: wait for every step before issuing the next.
    for (int i = 0; i < 5; ++i) must(model.Step(b, &logits), "warmup");
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < kSteps; ++i) must(model.Step(b, &logits), "step");
    const auto t1 = std::chrono::steady_clock::now();

    // Device sampling but still serialized: issue, then immediately wait. This
    // isolates what on-device sampling is worth on its own -- it removes a
    // 304 KB logits copy per token in favour of a 4-byte one -- from what the
    // overlap adds on top. Without this row the two would be reported as one
    // number and credited to the pipeline.
    b.tokens_from_device = true;
    for (int i = 0; i < 5; ++i) {
      (void)model.StepAsync(b);
      (void)model.AwaitStep(&tokens);
    }

    // Interleaved, per-step minima. Two facts forced this shape.
    //
    // First, measuring each configuration as one 60-step block put a 35% drift
    // between the first block and the rest -- running the *same* loop at both
    // ends produced 5.90 and 7.98 ms/token. Bulk wall-clock over sustained
    // generation measures the machine warming up as much as the code.
    //
    // Second, and more importantly: the two arrangements below issue the same
    // sequence of calls. `Await(i); StepAsync(i+1)` never has two steps in
    // flight, so it is not a pipeline -- it is the serialized loop with the
    // issue and the wait written in the other order. Real depth-1 overlap needs
    // step N+1 enqueued *before* step N is awaited, which needs a per-step
    // event and result buffer rather than the single pair this holds. That is
    // not built, and reporting these two as "serialized vs overlapped" would
    // claim it was.
    //
    // So what is compared is what actually differs: where sampling happens.
    double best_host = 1e9, best_gpu = 1e9;

    for (int round = 0; round < 12; ++round) {
      {
        b.tokens_from_device = false;
        const auto a0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) (void)model.Step(b, &logits);
        const auto a1 = std::chrono::steady_clock::now();
        best_host = std::min(
            best_host,
            std::chrono::duration<double, std::milli>(a1 - a0).count() / 5);
      }
      {
        b.tokens_from_device = true;
        const auto a0 = std::chrono::steady_clock::now();
        for (int i = 0; i < 5; ++i) {
          (void)model.StepAsync(b);
          (void)model.AwaitStep(&tokens);
        }
        const auto a1 = std::chrono::steady_clock::now();
        best_gpu = std::min(
            best_gpu,
            std::chrono::duration<double, std::milli>(a1 - a0).count() / 5);
      }
    }

    const double sync_ms = best_host;
    const double sampled_ms = best_gpu;

    std::printf(
        "\nsustained generation, batch 1 (interleaved, per-burst minima):\n");
    std::printf("  host sampling  %.4f ms/token  %6.1f tok/s\n", sync_ms,
                1000.0 / sync_ms);
    std::printf("  gpu sampling   %.4f ms/token  %6.1f tok/s   %.2fx\n",
                sampled_ms, 1000.0 / sampled_ms, sync_ms / sampled_ms);
  }

  return failures == 0 ? 0 : 1;
}

}  // namespace
}  // namespace inferx::bench

int main(int argc, char** argv) { return inferx::bench::Main(argc, argv); }
