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

  /// \brief Convenience: parse the config and open the checkpoint in one step.
  static StatusOr<Qwen2Model> LoadFromDirectory(std::string_view dir);

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

  const ModelConfig& config() const;

  /// \brief Device bytes held by the weights.
  size_t WeightBytes() const;

 private:
  struct Impl;

  explicit Qwen2Model(std::unique_ptr<Impl> impl);

  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::model
