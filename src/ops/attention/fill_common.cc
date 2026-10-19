#include "inferx/ops/attention/builder.h"

#include <algorithm>
#include <cstring>
#include <vector>

#include "inferx/core/device_runtime.h"

namespace inferx::attn {
namespace {

/// Enqueues one host-to-device copy of an int32 array.
Status Upload(ops::OpContext& ctx, const DeviceBuffer& buffer,
              const std::vector<std::int32_t>& values) {
  const std::size_t bytes = values.size() * sizeof(std::int32_t);
  if (bytes > buffer.size()) {
    return InternalError("metadata buffer too small: need ", bytes, " have ", buffer.size());
  }
  return ctx.Runtime().Memory().CopyAsync(buffer.data(), values.data(), bytes,
                                          CopyKind::kHostToDevice, ctx.GetStream());
}

}  // namespace

/// \brief The common paged-KV metadata builder shared by all backends
///        (composition over the FillCommonMetadata free function).
class CommonMetadataBuilder final : public MetadataBuilder {
 public:
  explicit CommonMetadataBuilder(GraphSupport graph) : graph_(graph) {}

  [[nodiscard]] GraphSupport GetGraphSupport() const noexcept override { return graph_; }
  [[nodiscard]] std::int32_t DecodeThreshold() const noexcept override { return 1; }

  [[nodiscard]] Status Build(const BatchDesc& desc, std::uint16_t group, BuildContext& ctx,
                             GroupMeta& out) noexcept override {
    return FillCommonMetadata(desc, group, ctx, out);
  }

  [[nodiscard]] Status BuildForCapture(const CaptureShape& shape, std::uint16_t group,
                                       BuildContext& ctx, GroupMeta& out) noexcept override {
    BatchDesc desc;
    desc.num_reqs = shape.num_reqs;
    desc.num_tokens = shape.NumTokens();
    desc.num_decodes = shape.num_reqs;
    desc.num_decode_tokens = desc.num_tokens;
    desc.max_query_len = shape.query_len;
    desc.max_seq_len = shape.query_len;
    desc.uniform_query_len = true;
    return FillCommonMetadata(desc, group, ctx, out);
  }

 private:
  GraphSupport graph_;
};

std::unique_ptr<MetadataBuilder> MakeCommonBuilder(GraphSupport graph) {
  return std::make_unique<CommonMetadataBuilder>(graph);
}

Status FillCommonMetadata(const BatchDesc& desc, std::uint16_t group, BuildContext& ctx,
                          GroupMeta& out) noexcept {
  if (ctx.ctx == nullptr || ctx.buffers == nullptr) {
    return InvalidArgumentError("FillCommonMetadata needs a context and buffers");
  }
  ops::OpContext& op_ctx = *ctx.ctx;
  DeviceMetaBuffers& buffers = *ctx.buffers;

  const std::int32_t num_reqs = desc.num_reqs;
  const std::int32_t block_size = desc.block_size > 0 ? desc.block_size : 16;

  // ---- shared, batch-wide arrays -------------------------------------------
  std::vector<std::int32_t> qo_indptr(static_cast<size_t>(num_reqs) + 1, 0);
  std::vector<std::int32_t> batch_indices(static_cast<size_t>(desc.num_tokens), 0);
  std::vector<std::int32_t> positions(static_cast<size_t>(desc.num_tokens), 0);
  std::int32_t token = 0;
  for (std::int32_t r = 0; r < num_reqs; ++r) {
    const std::int32_t len = desc.query_lens[static_cast<size_t>(r)];
    qo_indptr[static_cast<size_t>(r) + 1] = token + len;
    for (std::int32_t j = 0; j < len; ++j, ++token) {
      batch_indices[static_cast<size_t>(token)] = r;
      positions[static_cast<size_t>(token)] = desc.seq_lens[static_cast<size_t>(r)] - len + j;
    }
  }
  if (token != desc.num_tokens) {
    return InvalidArgumentError("query lens sum to ", token, " but num_tokens is ",
                                desc.num_tokens);
  }
  INFERX_RETURN_IF_ERROR(Upload(op_ctx, buffers.qo_indptr, qo_indptr));
  INFERX_RETURN_IF_ERROR(Upload(op_ctx, buffers.batch_indices, batch_indices));
  INFERX_RETURN_IF_ERROR(Upload(op_ctx, buffers.positions, positions));

  // ---- per-group page tables ------------------------------------------------
  const std::span<const BlockId> table = desc.block_ids[group];
  std::vector<std::int32_t> kv_indptr(static_cast<size_t>(num_reqs) + 1, 0);
  std::vector<std::int32_t> last_page_len(static_cast<size_t>(num_reqs), 0);
  std::vector<std::int32_t> kv_indices;
  std::int32_t pages_seen = 0;
  for (std::int32_t r = 0; r < num_reqs; ++r) {
    const std::int32_t seq_len = desc.seq_lens[static_cast<size_t>(r)];
    const std::int32_t pages =
        static_cast<std::int32_t>((seq_len + block_size - 1) / block_size);
    last_page_len[static_cast<size_t>(r)] = ((seq_len - 1) % block_size) + 1;
    kv_indptr[static_cast<size_t>(r) + 1] = pages_seen + pages;
    for (std::int32_t p = 0; p < pages; ++p, ++pages_seen) {
      kv_indices.push_back(static_cast<std::int32_t>(table[static_cast<size_t>(pages_seen)]));
    }
  }
  if (pages_seen > desc.block_ids_num_pages[group]) {
    return InvalidArgumentError("group ", group, " page table holds ", pages_seen,
                                " pages but the batch declares ",
                                desc.block_ids_num_pages[group]);
  }
  INFERX_RETURN_IF_ERROR(Upload(op_ctx, buffers.kv_indptr, kv_indptr));
  INFERX_RETURN_IF_ERROR(Upload(op_ctx, buffers.kv_indices, kv_indices));
  INFERX_RETURN_IF_ERROR(Upload(op_ctx, buffers.last_page_len, last_page_len));

  out = GroupMeta{};
  out.qo_indptr = reinterpret_cast<const std::int32_t*>(buffers.qo_indptr.data());
  out.kv_indptr = reinterpret_cast<const std::int32_t*>(buffers.kv_indptr.data());
  out.kv_indices = reinterpret_cast<const std::int32_t*>(buffers.kv_indices.data());
  out.last_page_len = reinterpret_cast<const std::int32_t*>(buffers.last_page_len.data());
  out.batch_indices = reinterpret_cast<const std::int32_t*>(buffers.batch_indices.data());
  out.positions = reinterpret_cast<const std::int32_t*>(buffers.positions.data());
  out.block_size = block_size;
  out.num_reqs = num_reqs;
  out.num_tokens = desc.num_tokens;
  out.num_decodes = desc.num_decodes;
  out.num_decode_tokens = desc.num_decode_tokens;
  out.num_prefills = desc.num_prefills;
  out.num_prefill_tokens = desc.num_prefill_tokens;
  out.max_query_len = desc.max_query_len;
  out.max_seq_len = desc.max_seq_len;
  out.shape = desc.Shape();
  return OkStatus();
}

}  // namespace inferx::attn
