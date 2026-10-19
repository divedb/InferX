#include "inferx/ops/attention/backends/reference.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <vector>

namespace inferx::attn {
namespace {

// ---- bf16 <-> float (host); the tree stores bf16 as raw uint16 -------------

float Bf16ToFloat(std::uint16_t h) {
  const std::uint32_t bits = static_cast<std::uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof(f));
  return f;
}

std::uint16_t FloatToBf16(float f) {
  std::uint32_t bits;
  std::memcpy(&bits, &f, sizeof(bits));
  // Round to nearest even on the dropped half.
  const std::uint32_t rounding = ((bits >> 16) & 1) + 0x7fff;
  return static_cast<std::uint16_t>((bits + rounding) >> 16);
}

/// Reads element i of a [rows, width] activation row in the spec's dtype,
/// widened to float.
float ReadActivation(const std::byte* row, std::int64_t i, DataType dtype) {
  switch (dtype) {
    case DataType::kBFloat16: {
      std::uint16_t h;
      std::memcpy(&h, row + i * 2, sizeof(h));
      return Bf16ToFloat(h);
    }
    case DataType::kFloat32: {
      float f;
      std::memcpy(&f, row + i * 4, sizeof(f));
      return f;
    }
    default:
      return 0.0f;  // Capability excludes other dtypes.
  }
}

void WriteActivation(std::byte* row, std::int64_t i, DataType dtype, float value) {
  switch (dtype) {
    case DataType::kBFloat16: {
      const std::uint16_t h = FloatToBf16(value);
      std::memcpy(row + i * 2, &h, sizeof(h));
      return;
    }
    case DataType::kFloat32: {
      std::memcpy(row + i * 4, &value, sizeof(value));
      return;
    }
    default:
      return;
  }
}

std::int64_t ActivationElementSize(DataType dtype) {
  return dtype == DataType::kFloat32 ? 4 : 2;
}

/// First key position (in the sequence's coordinate space) that the query
/// at `pos` attends to under the mask.
std::int64_t FirstNeeded(std::int64_t pos, const MaskSpec& mask) {
  switch (mask.kind) {
    case MaskKind::kCausal:
    case MaskKind::kBidirectional:
      return 0;
    case MaskKind::kSlidingWindow:
      return std::max<std::int64_t>(0, pos - mask.window + 1);
    case MaskKind::kChunked:
      return (pos / mask.window) * mask.window;
  }
  return 0;
}

/// Naive but exact attention for one dense layer, on host memory. Reads
/// the pool's [2, block_tokens, Hkv, D] pages directly from the CacheView.
class ReferenceKernel final : public Kernel {
 public:
  ReferenceKernel(AttentionSpec spec) : spec_(std::move(spec)) {}

  [[nodiscard]] Status Run(const RunArgs& args) const noexcept override {
    if (args.ctx.GetDevice().Type() != DeviceType::kCpu) {
      return UnimplementedError("the reference backend runs on CPU devices only");
    }
    const auto* dense = std::get_if<DenseParams>(&spec_.params);
    if (dense == nullptr) {
      return InternalError("reference kernel built for a non-dense spec");
    }
    const GroupMeta& meta = args.meta;
    const Inputs& in = args.in;
    const std::int64_t tokens = meta.num_tokens;
    const std::int64_t q_heads = dense->num_q_heads;
    const std::int64_t kv_heads = dense->num_kv_heads;
    const std::int64_t head_dim = dense->head_dim;
    const std::int64_t v_dim = dense->v_head_dim != 0 ? dense->v_head_dim : head_dim;
    const std::int64_t group = q_heads / kv_heads;
    const float scale =
        spec_.softmax_scale != 0.0f ? spec_.softmax_scale : 1.0f / std::sqrt(float(head_dim));

    if (in.q.rank != 3 || in.q.Dim(1) != q_heads || in.q.Dim(2) != head_dim ||
        in.q.Dim(0) != tokens) {
      return InvalidArgumentError("reference kernel: q must be [tokens, q_heads, head_dim]");
    }
    if (args.out.rank != 3 || args.out.Dim(1) != q_heads || args.out.Dim(2) != v_dim ||
        args.out.Dim(0) != tokens) {
      return InvalidArgumentError("reference kernel: out must be [tokens, q_heads, v_dim]");
    }

    // Fused KV write into the pool pages (skip when this layer has no
    // cache: encoder-self attention over the step's own k/v).
    const bool has_cache = args.cache.base != nullptr;
    if (in.k.Defined()) {
      if (in.k.rank != 3 || in.k.Dim(1) != kv_heads || in.k.Dim(2) != head_dim ||
          in.v.rank != 3 || in.v.Dim(1) != kv_heads || in.v.Dim(2) != v_dim) {
        return InvalidArgumentError("reference kernel: k/v must be [tokens, kv_heads, dim]");
      }
      INFERX_RETURN_IF_ERROR(
          WriteKv(args, *dense, tokens, kv_heads, head_dim, v_dim, has_cache));
    }

    const std::int64_t q_elem = ActivationElementSize(spec_.q_dtype);
    const std::int64_t out_elem = ActivationElementSize(spec_.q_dtype);
    const std::int64_t kv_elem =
        ActivationElementSize(has_cache ? spec_.kv_dtype : spec_.q_dtype);
    // Page layout [2, block_tokens, Hkv, D]: K plane first, then V.
    const std::int64_t plane = meta.block_size * kv_heads;
    std::vector<float> scores;

    for (std::int64_t t = 0; t < tokens; ++t) {
      const std::int32_t seq = meta.batch_indices[t];
      const std::int32_t pos = meta.positions[t];
      const std::int32_t pages = meta.kv_indptr[seq + 1] - meta.kv_indptr[seq];
      const std::int64_t kv_len =
          has_cache ? (static_cast<std::int64_t>(pages - 1) * meta.block_size +
                       meta.last_page_len[seq])
                    : /* stateless: attend within the step */ pos + 1;
      std::int64_t lo = FirstNeeded(pos, spec_.mask);
      if (!has_cache && spec_.mask.kind == MaskKind::kBidirectional) lo = 0;

      for (std::int64_t h = 0; h < q_heads; ++h) {
        const std::int64_t kv_h = h / group;
        scores.clear();
        float max_score = -std::numeric_limits<float>::infinity();
        for (std::int64_t j = lo; j <= pos && j < kv_len; ++j) {
          std::int64_t page = 0, slot = j;
          if (has_cache) {
            page = meta.kv_indices[meta.kv_indptr[seq] + j / meta.block_size];
            if (page == KvBlockPool::kNullBlockId) continue;  // rolled off
            slot = j % meta.block_size;
          }
          const std::byte* k_row = has_cache
                                       ? args.cache.Page(page) +
                                             (slot * kv_heads + kv_h) * head_dim * kv_elem
                                       : KeyRowStep(args.in, *dense, j, kv_h, kv_elem);
          float dot = 0.0f;
          const std::byte* q_row = static_cast<const std::byte*>(in.q.data) +
                                   t * in.q.Stride(0) * q_elem + h * in.q.Stride(1) * q_elem;
          for (std::int64_t d = 0; d < head_dim; ++d) {
            dot += ReadActivation(q_row + d * in.q.Stride(2) * q_elem, 0, spec_.q_dtype) *
                   ReadActivation(k_row, d, spec_.kv_dtype);
          }
          float score = dot * scale;
          if (spec_.features.Has(Feature::kSoftCap) && spec_.softcap > 0.0f) {
            score = spec_.softcap * std::tanh(score / spec_.softcap);
          }
          scores.push_back(score);
          max_score = std::max(max_score, score);
        }
        // Softmax and value accumulation over the surviving keys; the
        // loop repeats the mask walk so no index bookkeeping is shared
        // between the two passes.
        std::byte* out_row = static_cast<std::byte*>(args.out.data) +
                             (t * q_heads + h) * v_dim * out_elem;
        for (std::int64_t d = 0; d < v_dim; ++d) {
          WriteActivation(out_row, d, spec_.q_dtype, 0.0f);
        }
        if (scores.empty()) continue;
        float denom = 0.0f;
        for (float& s : scores) {
          s = std::exp(s - max_score);
          denom += s;
        }
        std::int64_t surviving = 0;
        for (std::int64_t j = lo; j <= pos && j < kv_len; ++j) {
          std::int64_t page = 0, slot = j;
          if (has_cache) {
            page = meta.kv_indices[meta.kv_indptr[seq] + j / meta.block_size];
            if (page == KvBlockPool::kNullBlockId) continue;
            slot = j % meta.block_size;
          }
          const float weight = scores[static_cast<size_t>(surviving)] / denom;
          ++surviving;
          const std::byte* v_row = has_cache
                                       ? args.cache.Page(page) +
                                             (plane + slot * kv_heads + kv_h) * v_dim * kv_elem
                                       : ValueRowStep(args.in, *dense, j, kv_h, v_dim, kv_elem);
          for (std::int64_t d = 0; d < v_dim; ++d) {
            const float acc = ReadActivation(out_row, d, spec_.q_dtype);
            WriteActivation(out_row, d, spec_.q_dtype,
                            acc + weight * ReadActivation(v_row, d, spec_.kv_dtype));
          }
        }
      }
    }
    return OkStatus();
  }

  [[nodiscard]] KvUpdateMode GetKvUpdateMode() const noexcept override {
    return KvUpdateMode::kFused;
  }
  [[nodiscard]] std::size_t WorkspaceBytes(const EngineLimits&) const noexcept override {
    return 0;  // Scratch is a stack vector per token.
  }

 private:
  /// K row of token j (step-local, cache-less layers).
  static const std::byte* KeyRowStep(const Inputs& in, const DenseParams& dense,
                                     std::int64_t j, std::int64_t kv_h, std::int64_t elem) {
    return static_cast<const std::byte*>(in.k.data) +
           (j * dense.num_kv_heads + kv_h) * dense.head_dim * elem;
  }
  static const std::byte* ValueRowStep(const Inputs& in, const DenseParams& dense,
                                       std::int64_t j, std::int64_t kv_h, std::int64_t v_dim,
                                       std::int64_t elem) {
    return static_cast<const std::byte*>(in.v.data) +
           (j * dense.num_kv_heads + kv_h) * v_dim * elem;
  }

  /// Writes this step's k/v into the layer's cache pages (fused update).
  Status WriteKv(const RunArgs& args, const DenseParams& dense, std::int64_t tokens,
                 std::int64_t kv_heads, std::int64_t head_dim, std::int64_t v_dim,
                 bool has_cache) const noexcept {
    const GroupMeta& meta = args.meta;
    const std::int64_t elem = ActivationElementSize(spec_.kv_dtype);
    const std::int64_t plane = meta.block_size * kv_heads;
    for (std::int64_t t = 0; t < tokens; ++t) {
      const std::int32_t seq = meta.batch_indices[t];
      const std::int32_t pos = meta.positions[t];
      if (!has_cache) break;  // Stateless layer: nothing to persist.
      const std::int64_t page =
          meta.kv_indices[meta.kv_indptr[seq] + pos / meta.block_size];
      if (page == KvBlockPool::kNullBlockId) {
        return InternalError("reference kernel: token ", t, " maps to the null block");
      }
      const std::int64_t slot = pos % meta.block_size;
      std::byte* k_dst = args.cache.Page(page) + (slot * kv_heads) * head_dim * elem;
      std::byte* v_dst =
          args.cache.Page(page) + (plane + slot * kv_heads) * v_dim * elem;
      const std::int64_t k_src = t * args.in.k.Stride(0);
      const std::int64_t v_src = t * args.in.v.Stride(0);
      for (std::int64_t h = 0; h < kv_heads; ++h) {
        for (std::int64_t d = 0; d < head_dim; ++d) {
          WriteActivation(k_dst, h * head_dim + d, spec_.kv_dtype,
                          ReadActivation(static_cast<const std::byte*>(args.in.k.data) +
                                             (k_src + h * args.in.k.Stride(1) +
                                              d * args.in.k.Stride(2)) * ActivationElementSize(spec_.q_dtype),
                                         0, spec_.q_dtype));
        }
        for (std::int64_t d = 0; d < v_dim; ++d) {
          WriteActivation(v_dst, h * v_dim + d, spec_.kv_dtype,
                          ReadActivation(static_cast<const std::byte*>(args.in.v.data) +
                                             (v_src + h * args.in.v.Stride(1) +
                                              d * args.in.v.Stride(2)) * ActivationElementSize(spec_.q_dtype),
                                         0, spec_.q_dtype));
        }
      }
    }
    return OkStatus();
  }

  AttentionSpec spec_;
};

constexpr Capability kReferenceCapability{
    .name = "reference",
    .mechanisms = ReferenceBackend::kMechanisms,
    .roles = {Role::kDecoderSelf, Role::kEncoderSelf, Role::kCross},
    .masks = {MaskKind::kCausal, MaskKind::kBidirectional, MaskKind::kSlidingWindow,
              MaskKind::kChunked},
    .q_dtypes = {DataType::kFloat32, DataType::kBFloat16},
    .kv_dtypes = {DataType::kFloat32, DataType::kBFloat16},
    .layouts = EnumSet<KvPageLayout>{KvPageLayout::kKvPlanarNhd},
    .phases = {.decode_only = true, .prefill_only = true, .mixed = true},
    .kv_update = KvUpdateMode::kFused,
    .graph = GraphSupport::kNever,
    .selectable_only_if_forced = true,
};

}  // namespace

const Capability& ReferenceBackend::GetCapability() const noexcept {
  return kReferenceCapability;
}

StatusOr<std::unique_ptr<Kernel>> ReferenceBackend::CreateKernel(
    const AttentionSpec& spec, const CacheView& view, const EngineLimits& limits) const {
  if (!std::holds_alternative<DenseParams>(spec.params)) {
    return InvalidArgumentError("reference backend implements the dense family only");
  }
  if (limits.device.Type() != DeviceType::kCpu) {
    return UnimplementedError("reference backend runs on CPU devices only");
  }
  if (view.base != nullptr && view.layout != KvPageLayout::kKvPlanarNhd) {
    return InvalidArgumentError("reference backend reads the kKvPlanarNhd layout");
  }
  return std::make_unique<ReferenceKernel>(spec);
}

StatusOr<std::unique_ptr<MetadataBuilder>> ReferenceBackend::CreateBuilder(
    const AttentionSpec&, const EngineLimits&) const {
  return MakeCommonBuilder(GraphSupport::kNever);
}

}  // namespace inferx::attn
