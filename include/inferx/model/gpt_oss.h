#pragma once

#include <cstdint>
#include <memory>
#include <string_view>
#include <vector>

#include "inferx/core/status.h"
#include "inferx/model/config.h"
#include "inferx/model/safetensors.h"

namespace inferx::model {

/// \brief gpt-oss-20b, forward only, correct rather than fast.
///
/// M11 Phase 2's deliverable, and it is deliberately the same kind of object
/// M2's `Qwen2Model::Forward` was: batch 1, no KV cache, full recompute of
/// every position on every call, synchronizing before it returns. It exists so
/// that its logits can be compared against HuggingFace's, which is the only way
/// to know that the MXFP4 decode, the routing, the sinks, the sliding window,
/// YaRN and the clamped activation are all right *together* rather than
/// individually.
///
/// **The memory strategy is the interesting part.** gpt-oss's expert weights
/// are 10.1 GB as MXFP4 and ~38 GB dequantized, so they cannot all be bf16 on
/// a 16 GB card — and this class does not have a 4-bit GEMM to avoid that with.
/// So the experts stay **packed on the host**, and each layer is uploaded and
/// dequantized into one reusable 1.6 GB scratch as it is reached:
///
///     per layer: upload 423 MB of MXFP4  ->  dequantize to 1.59 GB bf16
///                run the layer            ->  reuse the same scratch
///
/// Peak device usage is about 5.5 GB, and a forward pass moves ~10 GB over
/// PCIe. That is roughly a second per call and completely acceptable for
/// something run once against a reference. It is *not* a serving path and must
/// not be mistaken for one: Phase 4's job is a mainloop that reads MXFP4
/// directly, at which point this class stays exactly as it is, as the thing
/// that says the fast path is right.
class GptOssModel {
 public:
  /// \brief Loads from a checkpoint directory.
  ///
  /// The non-expert weights (attention, norms, router, embedding, lm_head —
  /// 3.6 GB) go to the device once. The expert weights are left in the
  /// checkpoint's mapping and read per layer per call.
  static StatusOr<GptOssModel> Load(std::string_view dir);

  ~GptOssModel();
  GptOssModel(const GptOssModel&) = delete;
  GptOssModel& operator=(const GptOssModel&) = delete;
  GptOssModel(GptOssModel&&) noexcept;
  GptOssModel& operator=(GptOssModel&&) noexcept;

  const ModelConfig& config() const;

  /// \brief Runs the stack over `token_ids` and returns every position's logits.
  ///
  /// \param token_ids  The prompt. Positions are `0..n-1`.
  /// \param out_logits Receives `[tokens × vocab]` fp32, row-major. fp32
  ///                   because the caller is a comparison against a reference,
  ///                   even though the stack ran in bf16.
  Status Forward(const std::vector<int32_t>& token_ids,
                 std::vector<float>* out_logits);

 private:
  struct Impl;
  explicit GptOssModel(std::unique_ptr<Impl> impl);
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::model
