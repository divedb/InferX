#include "inferx/models/layers/attention.h"

#include <utility>

namespace inferx {
namespace models {

StatusOr<Attention> Attention::Create(Geometry geometry) {
  ops::AttentionParams params;
  params.query_heads = geometry.num_heads;
  params.kv_heads = geometry.num_kv_heads;
  params.head_dim = geometry.head_dim;
  params.scale = geometry.scale;
  params.sliding_window = geometry.sliding_window;
  INFERX_RETURN_IF_ERROR(ops::ValidateAttentionGeometry(params));
  return Attention(geometry);
}

Status Attention::Forward(ops::OpContext& ctx, const AttentionMetadata& attn,
                          const KvBlockPool& kv, int64_t layer,
                          ops::AttentionPlanWorkspace& plan, const Tensor& positions,
                          const Tensor& q, const Tensor& k, const Tensor& v,
                          Tensor& out) const {
  INFERX_ASSIGN_OR_RETURN(Tensor key_cache, kv.KeyCache(layer));
  INFERX_ASSIGN_OR_RETURN(Tensor value_cache, kv.ValueCache(layer));

  // The cache append must complete before the attention read; both run on
  // the context's stream, so ordering is the stream's.
  INFERX_RETURN_IF_ERROR(ops::WritePagedKv(
      ctx, k, v, positions, *attn.batch_indices, *attn.kv_indptr, *attn.kv_indices,
      key_cache, value_cache, kv.BlockSize()));

  ops::AttentionParams params;
  params.query_heads = geometry_.num_heads;
  params.kv_heads = geometry_.num_kv_heads;
  params.head_dim = geometry_.head_dim;
  params.scale = geometry_.scale;
  params.sliding_window = geometry_.sliding_window;

  return ops::PagedAttention(ctx, q, *attn.qo_indptr, *attn.kv_indptr, *attn.kv_indices,
                             *attn.last_page_len, attn.host_qo_indptr, attn.num_seqs,
                             key_cache, value_cache, kv.BlockSize(), params, plan, out);
}

}  // namespace models
}  // namespace inferx
