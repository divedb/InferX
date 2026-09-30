#ifndef INFERX_SAMPLING_SAMPLING_KERNELS_H_
#define INFERX_SAMPLING_SAMPLING_KERNELS_H_

#include <cstdint>
#include <optional>

#include "inferx/core/status.h"
#include "inferx/core/tensor.h"
#include "inferx/ops/op_context.h"
#include "inferx/sampling/sampling_metadata.h"

namespace inferx::sampling::cuda {

/// \brief Device-resident per-row sampling parameters, packed SoA.
///
/// Uploaded once per step before any kernel launch so Sample() stays free
/// of allocations and host synchronization inside the stream.
struct DeviceParams {
  DeviceParams() = default;
  DeviceParams(const DeviceParams&) = delete;
  DeviceParams& operator=(const DeviceParams&) = delete;
  DeviceParams(DeviceParams&&) = default;
  DeviceParams& operator=(DeviceParams&&) = default;
  std::optional<Tensor> temperature;    ///< float32 [batch]
  std::optional<Tensor> top_k;          ///< int32 [batch]
  std::optional<Tensor> top_p;          ///< float32 [batch]
  std::optional<Tensor> min_p;          ///< float32 [batch]
  std::optional<Tensor> penalties;      ///< float32 [batch, 3] repetition/presence/frequency
  std::optional<Tensor> seeds;          ///< uint64 [batch] (engine default when unset)
  std::optional<Tensor> rng_offsets;    ///< uint64 [batch]
  std::optional<Tensor> greedy;         ///< int32 [batch]
  std::optional<Tensor> bias_ptr;       ///< int32 [batch + 1] CSR into bias_entries
  std::optional<Tensor> bias_entries;   ///< int64 [capacity] packed (token << 32 | bias bits)
  std::optional<Tensor> allow_ptr;      ///< int32 [batch + 1] CSR into allow_entries
  std::optional<Tensor> allow_entries;  ///< int32 [capacity]
  std::optional<Tensor> hist_ptr;       ///< int32 [batch + 1] CSR into hist_entries
  std::optional<Tensor> hist_entries;   ///< int32 [capacity] penalized token ids
};

/// \brief Runs the full sampling pipeline over [batch, vocab] logits.
///
/// Per row: logit bias, allowlist masking, repetition/presence/frequency
/// penalties, temperature, top-k / top-p / min-p filters (selection by
/// repeated max extraction), and a counter-based draw from the renormalized
/// distribution. One block per row; greedy rows degenerate to argmax with
/// the op's lowest-index tie break.
Status SampleRows(ops::OpContext& ctx, const Tensor& logits, const DeviceParams& params,
                  const Tensor& probs_workspace, Tensor& output);

}  // namespace inferx::sampling::cuda

#endif  // INFERX_SAMPLING_SAMPLING_KERNELS_H_
