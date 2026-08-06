#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "inferx/core/kv_cache.h"
#include "inferx/core/status.h"
#include "inferx/model/forward_batch.h"
#include "inferx/model/config.h"
#include "inferx/model/safetensors.h"

namespace inferx::comm {
class Communicator;
}

namespace inferx::model {

/// \brief A Qwen2/Llama decoder stack, resident on one GPU, running in bf16.
///
/// M2's scope exactly: batch 1, no KV cache, full recompute of every position
/// on every call. That is not a serving path and is not meant to be — it is the
/// implementation whose logits get compared against a reference, so it is
/// written to be checkable rather than fast. The KV cache and paged attention
/// arrive at M3, and the shapes here are already the ones those want.
///
/// Weights are uploaded once at `Load` and never move. A 3B model in bf16 is
/// about 6.2 GB, which is why the loader streams tensor by tensor from the
/// checkpoint's mapping rather than staging the whole file in host memory
/// first.
class Qwen2Model {
 public:
  /// \brief Loads a model from an open checkpoint onto the current CUDA device.
  ///
  /// \param config Parsed `config.json`, already validated.
  /// \param ckpt   The checkpoint. Only read during this call; the model does
  ///               not retain it, since every weight has been copied to the
  ///               device by the time it returns.
  /// \return       The model, or the first weight that was missing, misshaped,
  ///               or would not fit.
  static StatusOr<Qwen2Model> Load(const ModelConfig& config,
                                   const Checkpoint& ckpt);

  /// Loads one tensor-parallel rank. The model takes ownership of `comm`.
  /// All ranks must load the same checkpoint with communicators from the same
  /// world and execute identical forward calls in rank threads.
  static StatusOr<Qwen2Model> Load(const ModelConfig& config,
                                   const Checkpoint& ckpt,
                                   std::unique_ptr<comm::Communicator> comm);

  /// \brief Convenience: parse the config and open the checkpoint in one step.
  static StatusOr<Qwen2Model> LoadFromDirectory(std::string_view dir);
  static StatusOr<Qwen2Model> LoadFromDirectory(
      std::string_view dir, std::unique_ptr<comm::Communicator> comm);

  ~Qwen2Model();

  Qwen2Model(const Qwen2Model&) = delete;
  Qwen2Model& operator=(const Qwen2Model&) = delete;
  Qwen2Model(Qwen2Model&&) noexcept;
  Qwen2Model& operator=(Qwen2Model&&) noexcept;

  /// \brief Runs the full stack over `token_ids` and returns the logits.
  ///
  /// Positions are `0..n-1`; there is no cache to continue from. Synchronizes
  /// before returning, because the caller is a test or a reference comparison
  /// rather than a pipeline.
  ///
  /// \param token_ids The prompt. Must be non-empty and within the vocabulary.
  /// \param out_logits Receives `[tokens × vocab]` floats, row-major. Logits
  ///                   come back as fp32 because that is what a comparison
  ///                   against a reference wants, even though the stack ran in
  ///                   bf16.
  Status Forward(const std::vector<int32_t>& token_ids,
                 std::vector<float>* out_logits);

  /// \brief Logits for the final position only.
  ///
  /// The common case for a reference check and for sampling, and it avoids
  /// moving `tokens × 151936` floats over PCIe when only the last row matters.
  Status ForwardLastLogits(const std::vector<int32_t>& token_ids,
                           std::vector<float>* out_logits);

  /// \brief Allocates the paged KV cache. Required before `Step`.
  ///
  /// Separate from `Load` because the pool's size is a serving decision -- how
  /// much VRAM is left after the weights, and how many concurrent sequences
  /// that buys -- not a property of the model.
  ///
  /// \param num_blocks Blocks in the pool, shared by all sequences.
  /// \param block_size Tokens per block. 16 by default (T10).
  Status AttachKvCache(int64_t num_blocks, int64_t block_size = 16);

  /// \brief The pool, for the scheduler to allocate blocks from.
  ///
  /// The model owns the allocation because the executor owns device memory
  /// (§3.1); it never decides *who* gets a block, which is why this is handed
  /// out rather than wrapped.
  KvBlockPool* kv_pool();

  /// \brief Runs one step over a batch, reading and writing the KV cache.
  ///
  /// This is the path that makes the engine an engine: `batch` may hold a
  /// prefill, a set of decode steps, or both, and the model neither knows nor
  /// cares which. Requires `AttachKvCache`.
  ///
  /// \param batch      What to run. \see ForwardBatch.
  /// \param out_logits Receives `[logits_indices.size() × vocab]` fp32,
  ///                   row-major, in the order `logits_indices` lists.
  Status Step(const ForwardBatch& batch, std::vector<float>* out_logits);

  /// \brief Sizes the activation buffers for a batch of `max_tokens`.
  ///
  /// Must be called before `CaptureDecodeGraph` if anything larger than a
  /// decode batch will ever run, and this is not an optimization -- it is a
  /// correctness requirement. Capture records device *addresses*. A graph
  /// captured while the activations are sized for a four-token decode replays
  /// against those addresses forever, so the first prefill that grows the
  /// buffers leaves every captured graph reading memory that has been freed,
  /// and the model quietly emits garbage rather than failing.
  ///
  /// Buffers only ever grow, so reserving the largest batch up front is enough
  /// to make every later `EnsureCapacity` a no-op and every captured pointer
  /// stable.
  ///
  /// \param max_tokens Largest token count any future batch will carry.
  Status ReserveActivations(int64_t max_tokens);

  /// \brief Captures a CUDA graph for one decode shape.
  ///
  /// A decode step is ~400 launches -- 36 layers of seven GEMMs and half a
  /// dozen small kernels each -- and at small batch the benchmark in
  /// bench/attention_bench.cc shows the attention kernel itself is launch-bound
  /// rather than work-bound. Replaying one graph instead of issuing 400
  /// launches is what §5.2 and M6 are after.
  ///
  /// A graph fixes the step's *structure*: which kernels, at what dimensions,
  /// reading which addresses. Every value stays live -- token ids, positions,
  /// slots, block tables and sequence lengths are read from buffers that are
  /// rewritten before each replay -- so one graph serves a decode step forever,
  /// as long as the shape holds. That is why the block table lives in a fixed
  /// preallocated buffer (§6.2): so it can be updated in place without
  /// re-capture.
  ///
  /// Capturing is optional. Without it `Step` runs the same code
  /// launch-by-launch, and the two are required to produce identical output.
  ///
  /// \param num_seqs           Sequences in the batch. Decode is one token
  ///                           each, so this is also the token count.
  /// \param max_blocks_per_seq Block table width, which must match what the
  ///                           scheduler emits.
  Status CaptureDecodeGraph(int64_t num_seqs, int64_t max_blocks_per_seq);

  /// \brief How many decode shapes are currently captured.
  int64_t captured_graphs() const;

  /// \brief Requantizes the layer weights to FP8 e4m3, halving what a step
  ///        must stream.
  ///
  /// A decode step is bandwidth-bound on weights: 6.17 GB per token is 8.4 ms
  /// at this card's peak, which is the floor everything else has been optimized
  /// against. FP8 moves the floor rather than approaching it.
  ///
  /// **This changes the model's output.** Not by rounding, the way bf16 does,
  /// but by quantization: one scale per tensor, three mantissa bits. It is a
  /// serving mode, not an optimization, and it is opt-in for that reason. The
  /// bf16 path stays the reference and stays default.
  ///
  /// Embeddings and the LM head are left in bf16. They are the same tensor here
  /// (tied), the embedding lookup reads rows out of it directly, and quantizing
  /// it would mean dequantizing on every lookup for 10% of the weight bytes.
  ///
  /// Activations are quantized per step, dynamically, one scale per tensor.
  /// That is what cuBLASLt's scalar scale pointers accept and it needs no
  /// calibration pass.
  ///
  /// Irreversible: the bf16 weights are freed. Load the model again to get them
  /// back.
  Status QuantizeWeightsToF8();

  /// \brief True once `QuantizeWeightsToF8` has run.
  bool weights_are_f8() const;

  /// \brief Requantizes projection weights to symmetric per-group int4.
  ///
  /// Serving projections then use the fused W4A16 kernel. Embeddings, norms,
  /// and the tied LM head remain bf16. Must run before graph capture and is
  /// mutually exclusive with `QuantizeWeightsToF8`.
  Status QuantizeWeightsToInt4();

  /// \brief True once `QuantizeWeightsToInt4` has run.
  bool weights_are_int4() const;

  /// \brief Switches the KV cache to FP8 e4m3 before it is allocated.
  ///
  /// Must be called before `AttachKvCache`, which is when the pool's element
  /// type is fixed. Per-layer K/V dequant scales are frozen from the first
  /// warmup forward; the attention path then reads the cache through the FP8
  /// kernels (`RunFp8`/`PrefillFp8`). Like `QuantizeWeightsToF8` this changes
  /// the model's output and is opt-in.
  Status EnableFp8KvCache();

  /// \brief True once `EnableFp8KvCache` has run.
  bool kv_is_fp8() const;

  /// \brief Turns on device-side greedy sampling.
  ///
  /// With it on, a decode step samples its own next token and writes it into
  /// the buffer the following step reads, so the host never needs the value to
  /// keep issuing work. That is the precondition for `StepAsync`: §5.2's
  /// pipeline only exists if the loop can run ahead of its own results.
  ///
  /// \param max_rows Sampling rows per step, i.e. the largest batch that will
  ///                 be run. One row per sequence.
  Status EnableDeviceSampling(int64_t max_rows);

  /// \brief Issues a step and returns without waiting for it.
  ///
  /// §5.2 at depth 1. The step is enqueued and the call returns; the caller is
  /// free to prepare the next one while the GPU works. Nothing about the
  /// arithmetic changes -- `Step` is this followed immediately by `AwaitStep`,
  /// and T6 keeps that synchronous form as the reference the overlapped one is
  /// diffed against.
  ///
  /// Requires `EnableDeviceSampling`: without it the next step's token ids
  /// would have to come back through the host, which is the wait this removes.
  Status StepAsync(const ForwardBatch& batch);

  /// \brief Waits for the outstanding `StepAsync` and returns its tokens.
  ///
  /// \param out_tokens Receives one sampled token per logits row, in the order
  ///                   the batch requested them.
  Status AwaitStep(std::vector<int32_t>* out_tokens);

  /// \brief GPU time of the last `Step`, in milliseconds.
  ///
  /// Measured with events around the launch, so it covers the device work and
  /// nothing else. Subtracting it from the wall time of `Step` gives the host
  /// wrapper: index staging, the FlashInfer plan, and the logits copy back.
  /// That split is the only way to tell whether more of the step belongs inside
  /// a graph or whether what remains is simply the work.
  double last_step_device_ms() const;

  const ModelConfig& config() const;
  int tensor_parallel_rank() const;
  int tensor_parallel_size() const;

  /// Aborts the communication backend after an unrecoverable rank failure.
  Status AbortCommunicator();

  /// \brief Device bytes held by the weights.
  size_t WeightBytes() const;

 private:
  struct Impl;

  /// Rejects the bf16-only recompute path once weights are FP8.
  Status RequireBf16Weights() const;

  explicit Qwen2Model(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::model
