#include <cstdint>
#include <unordered_map>
#include <vector>

struct KvCacheConfig {
  int num_layers;
  int num_kv_heads;
  int head_dim;
  int block_size;
  int num_blocks;
};

struct KvLayerStorage {
  Tensor k;  /// [num_blocks, block_size, num_kv_heads, head_dim]
  Tensor v;  /// [num_blocks, block_size, num_kv_heads, head_dim]
};

class KvCacheStorage {
 public:
  KvCacheStorage(const KvCacheConfig& config);
  ~KvCacheStorage();

  /// \brief Returns the storage for a specific layer.
  ///
  /// \param layer The layer index.
  /// \return      The storage for that layer.
  KvLayerStorage& GetLayerStorage(int layer);

  /// \brief Returns the storage for a specific layer (const version).
  ///
  /// \param layer The layer index.
  /// \return      The storage for that layer.
  const KvLayerStorage& GetLayerStorage(int layer) const;

 private:
  std::vector<KvLayerStorage> layers_;
};

using BlockId = uint32_t;

struct KvBlock {
  BlockId id;
  uint32_t ref_count = 0;
};

class KvBlockPool {
 public:
  explicit KvBlockPool(int num_blocks);

  BlockId Acquire();
  void Retain(BlockId block_id);
  void Release(BlockId block_id);

  size_t NumFreeBlocks() const;

 private:
  std::vector<KvBlock> blocks_;
  std::vector<BlockId> free_blocks_;
};

using RequestId = uint64_t;

struct RequestKvState {
  RequestId request_id;

  std::vector<BlockId> block_ids;

  size_t num_tokens = 0;
};

class KvCacheManager {
 public:
  explicit KvCacheManager(const KvCacheConfig& config);

  // Ensure enough blocks exist for total_tokens.
  bool Reserve(RequestId request_id, size_t total_tokens);

  // Update token count after successful KV writes.
  void Commit(RequestId request_id, size_t num_tokens);

  // Release all blocks owned by the request.
  void Release(RequestId request_id);

  const RequestKvState& GetState(RequestId request_id) const;

 private:
  KvCacheConfig config_;
  KvCacheStorage storage_;
  KvBlockPool pool_;

  std::unordered_map<RequestId, RequestKvState> requests_;
};