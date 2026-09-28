#ifndef INFERX_SAMPLING_SAMPLER_H_
#define INFERX_SAMPLING_SAMPLER_H_

#include <memory>
#include <optional>

#include "inferx/core/device.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/execution_context.h"
#include "inferx/sampling/sampling_kernels.h"
#include "inferx/sampling/sampling_metadata.h"
#include "inferx/sampling/sampler_output.h"

namespace inferx::sampling {

/// \brief Executes the sampling pipeline over one step's logits.
///
/// Sample() turns a [batch, vocab] logits tensor plus the batch's
/// SamplingMetadata into one token per row. Greedy batches take the fused
/// argmax fast path; everything else runs the sampling pipeline (bias and
/// allowlist, repetition/presence/frequency penalties, temperature,
/// top-k/top-p/min-p, counter-based seeded draws) whose parameters are
/// mirrored to device buffers at the head of Sample().
///
/// CUDA contract: Sample() only launches kernels on the context's stream —
/// no allocations, no host synchronization — so it composes with continuous
/// batching loops and is safe to capture inside a CUDA graph. The workspace
/// (two-pass reduction partials and the result buffer) is allocated once in
/// Create() and its device addresses stay stable across steps.
class Sampler {
 public:
  /// \brief Allocates the workspace for batches up to `max_num_seqs` rows
  ///        over a `vocab_size` vocabulary on `device`.
  static StatusOr<std::unique_ptr<Sampler>> Create(int max_num_seqs,
                                                   std::int64_t vocab_size,
                                                   DeviceId device);

  /// \brief Samples one token per row of `logits`.
  ///
  /// \param ctx      Execution lane; work is enqueued on ctx.stream().
  /// \param logits   [batch, vocab] float32 or bfloat16 matrix.
  /// \param metadata Batch description, requests.size() == batch.
  /// \return         Tensors aliasing this sampler's workspace, valid until
  ///                 the next Sample call.
  StatusOr<SamplerOutput> Sample(ops::ExecutionContext& ctx, const Tensor& logits,
                                 const SamplingMetadata& metadata);

 private:
  Sampler(std::optional<Tensor> values, std::optional<Tensor> indices, Tensor results,
          std::int64_t vocab_size, std::optional<cuda::DeviceParams> params,
          std::optional<Tensor> probs, int64_t max_num_seqs);

  Status UploadParams(ops::ExecutionContext& ctx, const SamplingMetadata& metadata);

  StatusOr<SamplerOutput> SampleCpuReference(ops::ExecutionContext& ctx, const Tensor& logits,
                                             const SamplingMetadata& metadata, Tensor sampled);

  std::optional<Tensor> values_;   ///< float32 argmax partials, [max_num_seqs * parts]; CUDA only.
  std::optional<Tensor> indices_;  ///< int32 argmax partial indices, same shape; CUDA only.
  Tensor results_;                 ///< int32 sampled ids, [max_num_seqs].
  std::int64_t vocab_size_ = 0;
  std::optional<cuda::DeviceParams> params_;  ///< CUDA pipeline mirrors; CUDA only.
  std::optional<Tensor> probs_;               ///< [max_num_seqs, vocab] float32; CUDA only.
  int64_t max_num_seqs_ = 0;
};

}  // namespace inferx::sampling

#endif  // INFERX_SAMPLING_SAMPLER_H_
