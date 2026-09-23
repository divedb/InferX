#ifndef INFERX_SAMPLING_SAMPLER_OUTPUT_H_
#define INFERX_SAMPLING_SAMPLER_OUTPUT_H_

#include "inferx/core/tensor.h"

namespace inferx::sampling {

/// \brief Results of one sampling step, device-resident.
///
/// Staying on the device is the point: the engine decides when a
/// device-to-host copy (and its synchronization) is actually needed. The
/// tensors alias the sampler's fixed workspace, so an output is valid until
/// the next Sampler::Sample call on that sampler.
struct SamplerOutput {
  /// \brief Sampled token per batch row, int32 [batch].
  Tensor sampled_token_ids;

  /// \brief Log-probability of each sampled token, float [batch]; undefined
  ///        until the logprob path is implemented and requested.
  Tensor logprobs;

  /// \brief Top-k (token, logprob) pairs per row arrive here once logprobs
  ///        land; planned as CSR tensors, not a host container.

  bool IsDefined() const { return sampled_token_ids.IsDefined(); }
};

}  // namespace inferx::sampling

#endif  // INFERX_SAMPLING_SAMPLER_OUTPUT_H_
