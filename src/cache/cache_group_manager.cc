#include <algorithm>
#include <cstddef>
#include <utility>

#include "inferx/cache/cache_group_manager.h"

namespace inferx {
namespace {

constexpr BlockId kNull = KvBlockPool::kNullBlockId;

int64_t CeilDiv(int64_t a, int64_t b) { return a <= 0 ? 0 : (a + b - 1) / b; }

/// Real blocks of `table`, tail first: the order the pool wants so the free
/// queue evicts tails before shared prefixes.
std::vector<BlockId> RealBlocksTailFirst(absl::Span<const BlockId> table) {
  std::vector<BlockId> out;
  out.reserve(table.size());
  for (auto it = table.rbegin(); it != table.rend(); ++it) {
    if (*it != kNull) out.push_back(*it);
  }
  return out;
}

// ===========================================================================
// Token-wise managers
// ===========================================================================

/// \brief Everything the token-wise kinds share. A kind is defined by three
/// small hooks: which token is the first one the next token still needs (the
/// retention floor), how many leading sink blocks never roll off, and how many
/// tokens the group covers. Full attention, sliding window, chunked local and
/// cross-attention differ only in those.
class TokenWiseManager : public CacheGroupManager {
 public:
  TokenWiseManager(CacheGroupSpec group, KvBlockPool* pool)
      : group_(std::move(group)), pool_(pool), block_size_(pool->BlockSize()) {}

  const CacheGroupSpec& group() const final { return group_; }

  // -- hit search and adoption ----------------------------------------------

  CacheHit FindLongestCacheHit(absl::Span<const BlockHash> hashes,
                               int64_t max_tokens) const final {
    CacheHit hit;
    if (!group_.spec.PrefixCacheable() || max_tokens < block_size_) return hit;

    const int64_t nb =
        std::min<int64_t>(static_cast<int64_t>(hashes.size()), max_tokens / block_size_);

    // Full attention (and a window that never clips) needs the whole prefix,
    // so one miss ends the search; other kinds must look past misses.
    const bool prefix_only = FirstNeededToken(nb * block_size_) == 0;

    std::vector<BlockId> found(static_cast<size_t>(nb), kNull);
    std::vector<int64_t> run(static_cast<size_t>(nb) + 1, 0);  // cached run ending at k-1
    int64_t prefix_run = 0;                                    // leading cached run
    bool prefix_open = true;
    for (int64_t i = 0; i < nb; ++i) {
      const std::optional<BlockId> id =
          pool_->LookupCached(KeyFor(hashes[static_cast<size_t>(i)]));
      if (id) {
        found[static_cast<size_t>(i)] = *id;
        run[static_cast<size_t>(i) + 1] = run[static_cast<size_t>(i)] + 1;
        if (prefix_open) ++prefix_run;
      } else {
        prefix_open = false;
        if (prefix_only) break;
      }
    }

    // A hit of k blocks (L = k * block_size tokens) is usable when every block
    // the next token still needs is cached: the sink blocks, plus the blocks
    // from the one holding the first needed token up to k.
    const int64_t sink_blocks = NumSinkBlocks();
    for (int64_t k = nb; k >= 1; --k) {
      const int64_t first_window = std::min(FirstNeededToken(k * block_size_) / block_size_, k);
      const int64_t need_sinks = std::min(sink_blocks, k);
      const bool overlap = first_window <= sink_blocks;  // sinks and window meet: need [0, k)
      const bool ok =
          overlap ? prefix_run >= k
                  : prefix_run >= need_sinks && run[static_cast<size_t>(k)] >= k - first_window;
      if (!ok) continue;

      hit.blocks.assign(static_cast<size_t>(k), kNull);
      for (int64_t i = 0; i < k; ++i) {
        if (overlap || i < need_sinks || i >= first_window) {
          hit.blocks[static_cast<size_t>(i)] = found[static_cast<size_t>(i)];
        }
      }
      hit.num_tokens = k * block_size_;
      return hit;
    }
    return hit;
  }

  Status AdoptCached(RequestGroupState& req, const CacheHit& hit) final {
    auto* s = std::get_if<TokenWiseRequestState>(&req);
    if (s == nullptr) return WrongKind();
    if (!s->blocks.empty()) {
      return FailedPreconditionError("group ", group_.group_index,
                                     ": a hit can only be adopted into an empty request");
    }
    if (hit.num_tokens != static_cast<int64_t>(hit.blocks.size()) * block_size_) {
      return InvalidArgumentError("group ", group_.group_index, ": hit covers ", hit.num_tokens,
                                  " tokens but lists ", hit.blocks.size(), " blocks");
    }

    std::vector<BlockId> pinned;
    pinned.reserve(hit.blocks.size());
    for (const BlockId block : hit.blocks) {
      if (block == kNull) continue;
      const Status status = pool_->AcquireBlock(block);
      if (!status.ok()) {
        std::reverse(pinned.begin(), pinned.end());
        static_cast<void>(pool_->ReleaseBlocks(pinned));
        return status;
      }
      pinned.push_back(block);
    }

    s->blocks = hit.blocks;
    s->num_cached_tokens = hit.num_tokens;
    s->num_published_blocks = static_cast<int64_t>(hit.blocks.size());
    return OkStatus();
  }

  // -- sizing ---------------------------------------------------------------

  int64_t NumBlocksHeldAt(int64_t target_tokens, int64_t encoder_tokens) const final {
    const int64_t tokens = TokensCovered(target_tokens, encoder_tokens);
    const int64_t total = CeilDiv(tokens, block_size_);
    const int64_t sink_blocks = NumSinkBlocks();
    const int64_t first_live = std::max(sink_blocks, FirstNeededToken(tokens) / block_size_);
    return std::min(total, sink_blocks) + std::max<int64_t>(0, total - first_live);
  }

  int64_t NumSkippedTokens(int64_t num_computed_tokens) const final {
    return FirstNeededToken(num_computed_tokens);
  }

  int64_t NumNewBlocksNeeded(const RequestGroupState& req, const GrowthTarget& target,
                             const CacheHit* pending_hit) const final {
    const auto* s = std::get_if<TokenWiseRequestState>(&req);
    if (s == nullptr) return 0;

    const int64_t len = pending_hit != nullptr
                            ? static_cast<int64_t>(pending_hit->blocks.size())
                            : static_cast<int64_t>(s->blocks.size());
    const GrowthPlan plan = Plan(*s, target, len, /*allow_cow=*/pending_hit == nullptr);
    int64_t needed = plan.reals + (plan.cow ? 1 : 0);

    // Pinning a cached block that is idle in the free queue removes it from
    // the capacity the scheduler sees, so it costs a block too.
    if (pending_hit != nullptr) {
      for (const BlockId block : pending_hit->blocks) {
        if (block != kNull && pool_->RefCount(block) == 0) ++needed;
      }
    }
    return needed;
  }

  // -- growth ---------------------------------------------------------------

  Status ReleaseSkipped(RequestGroupState& req, int64_t num_computed_tokens) final {
    auto* s = std::get_if<TokenWiseRequestState>(&req);
    if (s == nullptr) return WrongKind();

    const int64_t hi = std::min<int64_t>(FirstNeededToken(num_computed_tokens) / block_size_,
                                         static_cast<int64_t>(s->blocks.size()));
    // Blocks wholly below the floor, never the sinks. Walk down and stop at
    // the first null: everything below it was rolled earlier.
    std::vector<BlockId> freed;
    for (int64_t i = hi - 1; i >= NumSinkBlocks(); --i) {
      BlockId& block = s->blocks[static_cast<size_t>(i)];
      if (block == kNull) break;
      freed.push_back(block);
      block = kNull;
    }
    return pool_->ReleaseBlocks(freed);  // highest index first: evicted first
  }

  StatusOr<Granted> Reserve(RequestGroupState& req, const GrowthTarget& target) final {
    auto* s = std::get_if<TokenWiseRequestState>(&req);
    if (s == nullptr) return WrongKind();
    INFERX_RETURN_IF_ERROR(CheckTarget(target));

    const GrowthPlan plan = Plan(*s, target, static_cast<int64_t>(s->blocks.size()), true);
    const int64_t n = plan.reals + (plan.cow ? 1 : 0);

    // Allocate first so a failure leaves the request untouched.
    INFERX_ASSIGN_OR_RETURN(auto fresh, pool_->AllocateBlocks(n));

    Granted granted;
    size_t next = 0;
    if (plan.cow) {
      const BlockId dst = fresh[next++];
      const size_t index = static_cast<size_t>(plan.cow_index);
      const BlockId src = s->blocks[index];
      // The child's reference on the shared block moves to the copy. Another
      // holder exists (ref > 1), so this cannot free it.
      const Status status = pool_->ReleaseBlock(src);
      if (!status.ok()) {
        std::reverse(fresh.begin(), fresh.end());
        static_cast<void>(pool_->ReleaseBlocks(fresh));
        return status;
      }
      s->blocks[index] = dst;
      granted.copies.push_back(BlockCopy{src, dst});
    }

    const int64_t sink_blocks = NumSinkBlocks();
    const int64_t floor_block = FirstNeededToken(target.num_computed_tokens) / block_size_;
    for (int64_t i = static_cast<int64_t>(s->blocks.size()); i < plan.total; ++i) {
      if (i >= sink_blocks && i < floor_block) {
        s->blocks.push_back(kNull);  // below the retention floor: keep indexing
      } else {
        const BlockId block = fresh[next++];
        s->blocks.push_back(block);
        granted.blocks.push_back(block);
      }
    }
    return granted;
  }

  Status Rollback(RequestGroupState& req, const Granted& granted) final {
    auto* s = std::get_if<TokenWiseRequestState>(&req);
    if (s == nullptr) return WrongKind();

    // Undo copy-on-write: the table goes back to the shared block.
    for (const BlockCopy& copy : granted.copies) {
      const auto it = std::find(s->blocks.rbegin(), s->blocks.rend(), copy.dst);
      if (it == s->blocks.rend()) {
        return InvalidArgumentError("group ", group_.group_index, ": copy target ", copy.dst,
                                    " is not in the request");
      }
      *it = copy.src;
      INFERX_RETURN_IF_ERROR(pool_->AcquireBlock(copy.src));
      INFERX_RETURN_IF_ERROR(pool_->ReleaseBlock(copy.dst));
    }

    // Drop everything appended from the first granted block on (null pads
    // included), then free the granted blocks. Pads before it can stay: the
    // floor only moves forward, so they remain correct.
    if (!granted.blocks.empty()) {
      const auto it = std::find(s->blocks.rbegin(), s->blocks.rend(), granted.blocks.front());
      if (it == s->blocks.rend()) {
        return InvalidArgumentError("group ", group_.group_index, ": granted block ",
                                    granted.blocks.front(), " is not in the request");
      }
      s->blocks.erase(it.base() - 1, s->blocks.end());
      const std::vector<BlockId> tail_first(granted.blocks.rbegin(), granted.blocks.rend());
      INFERX_RETURN_IF_ERROR(pool_->ReleaseBlocks(tail_first));
    }
    return OkStatus();
  }

  // -- publication and release ----------------------------------------------

  Status PublishCompleted(RequestGroupState& req, absl::Span<const BlockHash> hashes,
                          int64_t num_computed_tokens) final {
    auto* s = std::get_if<TokenWiseRequestState>(&req);
    if (s == nullptr) return WrongKind();
    if (!group_.spec.PrefixCacheable()) return OkStatus();

    const int64_t full = std::min<int64_t>({num_computed_tokens / block_size_,
                                            static_cast<int64_t>(hashes.size()),
                                            static_cast<int64_t>(s->blocks.size())});
    for (int64_t i = s->num_published_blocks; i < full; ++i) {
      const BlockId block = s->blocks[static_cast<size_t>(i)];
      if (block == kNull) continue;  // rolled off before it could be published
      pool_->PublishBlock(KeyFor(hashes[static_cast<size_t>(i)]), block);
    }
    s->num_published_blocks = std::max(s->num_published_blocks, full);
    return OkStatus();
  }

  Status Release(RequestGroupState& req) final {
    auto* s = std::get_if<TokenWiseRequestState>(&req);
    if (s == nullptr) return WrongKind();
    const Status status = pool_->ReleaseBlocks(RealBlocksTailFirst(s->blocks));
    req = TokenWiseRequestState{};
    return status;
  }

  Status Fork(const RequestGroupState& parent, int64_t num_children,
              std::vector<RequestGroupState>* children,
              std::vector<SlotCopy>* /*slot_copies*/) final {
    const auto* p = std::get_if<TokenWiseRequestState>(&parent);
    if (p == nullptr) return WrongKind();
    if (children == nullptr || num_children <= 0) {
      return InvalidArgumentError("group ", group_.group_index, ": cannot fork into ",
                                  num_children, " children");
    }

    std::vector<TokenWiseRequestState> made;
    made.reserve(static_cast<size_t>(num_children));
    const auto undo = [&] {
      for (auto it = made.rbegin(); it != made.rend(); ++it) {
        static_cast<void>(pool_->ReleaseBlocks(RealBlocksTailFirst(it->blocks)));
      }
    };

    for (int64_t c = 0; c < num_children; ++c) {
      TokenWiseRequestState child = *p;  // same table, counters and publish cursor
      for (size_t i = 0; i < child.blocks.size(); ++i) {
        if (child.blocks[i] == kNull) continue;
        const Status status = pool_->AcquireBlock(child.blocks[i]);
        if (!status.ok()) {
          const std::vector<BlockId> partial(
              child.blocks.begin(), child.blocks.begin() + static_cast<std::ptrdiff_t>(i));
          static_cast<void>(pool_->ReleaseBlocks(RealBlocksTailFirst(partial)));
          undo();
          return status;
        }
      }
      made.push_back(std::move(child));
    }

    children->clear();
    for (TokenWiseRequestState& child : made) children->emplace_back(std::move(child));
    return OkStatus();
  }

 protected:
  /// First token position the next token (at `position`) still attends to,
  /// ignoring sinks. Tokens below it are skippable.
  virtual int64_t FirstNeededToken(int64_t /*position*/) const { return 0; }
  /// Leading blocks that are never rolled off.
  virtual int64_t NumSinkBlocks() const { return 0; }
  /// Tokens this group covers for a request.
  virtual int64_t TokensCovered(int64_t target_tokens, int64_t /*encoder_tokens*/) const {
    return target_tokens;
  }
  /// True when blocks are written once and then only read, so forks can share
  /// them without copy-on-write.
  virtual bool SharedBlocksAreImmutable() const { return false; }

  Status WrongKind() const {
    return InvalidArgumentError("request state does not match token-wise group ",
                                group_.group_index);
  }

  BlockKey KeyFor(const BlockHash& hash) const {
    BlockKey key;
    key.hash = hash;
    key.group_index = static_cast<int32_t>(group_.group_index);
    return key;
  }

  const CacheGroupSpec group_;
  KvBlockPool* const pool_;
  const int64_t block_size_;

 private:
  struct GrowthPlan {
    int64_t total = 0;  ///< Blocks the sequence spans at the target.
    int64_t pads = 0;   ///< New entries below the floor: null, no storage.
    int64_t reals = 0;  ///< New entries that need a block.
    bool cow = false;   ///< The block being written is shared: duplicate it.
    int64_t cow_index = -1;
  };

  GrowthPlan Plan(const TokenWiseRequestState& s, const GrowthTarget& target, int64_t len,
                  bool allow_cow) const {
    GrowthPlan plan;
    const int64_t tokens = TokensCovered(target.target_tokens, target.encoder_tokens);
    plan.total = CeilDiv(tokens, block_size_);

    const int64_t missing = std::max<int64_t>(0, plan.total - len);
    const int64_t floor_block = FirstNeededToken(target.num_computed_tokens) / block_size_;
    const int64_t lo = std::max(len, NumSinkBlocks());
    const int64_t hi = std::min(plan.total, floor_block);
    plan.pads = std::max<int64_t>(0, hi - lo);
    plan.reals = missing - plan.pads;

    const int64_t computed = target.num_computed_tokens;
    if (allow_cow && !SharedBlocksAreImmutable() && tokens > computed &&
        computed % block_size_ != 0) {
      const int64_t index = computed / block_size_;
      if (index < static_cast<int64_t>(s.blocks.size())) {
        const BlockId block = s.blocks[static_cast<size_t>(index)];
        if (block != kNull && pool_->RefCount(block) > 1) {
          plan.cow = true;
          plan.cow_index = index;
        }
      }
    }
    return plan;
  }

  static Status CheckTarget(const GrowthTarget& target) {
    if (target.num_computed_tokens < 0 || target.target_tokens < target.num_computed_tokens) {
      return InvalidArgumentError("bad growth target: computed=", target.num_computed_tokens,
                                  " target=", target.target_tokens);
    }
    return OkStatus();
  }
};

class FullAttentionManager final : public TokenWiseManager {
 public:
  using TokenWiseManager::TokenWiseManager;
};

/// Trailing window of W tokens plus optional leading sink tokens.
class SlidingWindowManager final : public TokenWiseManager {
 public:
  SlidingWindowManager(CacheGroupSpec group, KvBlockPool* pool)
      : TokenWiseManager(std::move(group), pool),
        window_(std::get<WindowRetention>(group_.spec.retention).window),
        sink_blocks_(std::get<WindowRetention>(group_.spec.retention).sink_tokens /
                     block_size_) {}

 protected:
  // The next token at `position` attends to the previous W - 1 tokens and
  // itself, so it needs positions from position - W + 1 on.
  int64_t FirstNeededToken(int64_t position) const override {
    return std::max<int64_t>(0, position - window_ + 1);
  }
  int64_t NumSinkBlocks() const override { return sink_blocks_; }

 private:
  const int64_t window_;
  const int64_t sink_blocks_;
};

/// Only the current chunk stays resident (Llama 4 chunked local attention).
class ChunkedLocalManager final : public TokenWiseManager {
 public:
  ChunkedLocalManager(CacheGroupSpec group, KvBlockPool* pool)
      : TokenWiseManager(std::move(group), pool),
        chunk_tokens_(std::get<ChunkRetention>(group_.spec.retention).chunk_tokens) {}

 protected:
  // The next token attends only inside its own chunk.
  int64_t FirstNeededToken(int64_t position) const override {
    return (position / chunk_tokens_) * chunk_tokens_;
  }

 private:
  const int64_t chunk_tokens_;
};

/// Cache sized by the encoder length; written once, then read-only. Never
/// prefix-cached (the spec says so), so hit search and publication are no-ops.
class CrossAttentionManager final : public TokenWiseManager {
 public:
  using TokenWiseManager::TokenWiseManager;

 protected:
  int64_t TokensCovered(int64_t /*target_tokens*/, int64_t encoder_tokens) const override {
    return encoder_tokens;
  }
  bool SharedBlocksAreImmutable() const override { return true; }
};

// ===========================================================================
// Recurrent manager
// ===========================================================================

/// \brief One state slot per request, in the state pool. Phase E: no state
/// prefix hits, so hit search and publication are no-ops. Counts returned by
/// the sizing methods are slots, not blocks: the facade compares them to the
/// state pool's free slots, never to the block pool's capacity.
class RecurrentStateManager final : public CacheGroupManager {
 public:
  RecurrentStateManager(CacheGroupSpec group, RecurrentStatePool* state_pool)
      : group_(std::move(group)), state_pool_(state_pool) {}

  const CacheGroupSpec& group() const override { return group_; }

  CacheHit FindLongestCacheHit(absl::Span<const BlockHash> /*hashes*/,
                               int64_t /*max_tokens*/) const override {
    return CacheHit{};
  }

  int64_t NumBlocksHeldAt(int64_t /*target_tokens*/,
                          int64_t /*encoder_tokens*/) const override {
    return 1;
  }

  Status AdoptCached(RequestGroupState& req, const CacheHit& hit) override {
    if (std::get_if<RecurrentRequestState>(&req) == nullptr) return WrongKind();
    if (hit.num_tokens != 0 || !hit.blocks.empty()) {
      return InvalidArgumentError("group ", group_.group_index,
                                  ": recurrent state cannot adopt a cached prefix yet");
    }
    return OkStatus();
  }

  int64_t NumNewBlocksNeeded(const RequestGroupState& req, const GrowthTarget& /*target*/,
                             const CacheHit* /*pending_hit*/) const override {
    const auto* s = std::get_if<RecurrentRequestState>(&req);
    return (s != nullptr && s->slot < 0) ? 1 : 0;
  }

  StatusOr<Granted> Reserve(RequestGroupState& req, const GrowthTarget& /*target*/) override {
    auto* s = std::get_if<RecurrentRequestState>(&req);
    if (s == nullptr) return WrongKind();

    Granted granted;
    if (s->slot >= 0) return granted;  // already holds its slot
    INFERX_ASSIGN_OR_RETURN(const int64_t slot, state_pool_->AllocateSlot());
    s->slot = slot;
    granted.slot = slot;
    return granted;
  }

  Status Rollback(RequestGroupState& req, const Granted& granted) override {
    auto* s = std::get_if<RecurrentRequestState>(&req);
    if (s == nullptr) return WrongKind();
    if (!granted.slot.has_value()) return OkStatus();
    if (s->slot != *granted.slot) {
      return InvalidArgumentError("group ", group_.group_index, ": slot ", *granted.slot,
                                  " is not the request's slot");
    }
    INFERX_RETURN_IF_ERROR(state_pool_->ReleaseSlot(s->slot));
    s->slot = -1;
    return OkStatus();
  }

  Status PublishCompleted(RequestGroupState& req, absl::Span<const BlockHash> /*hashes*/,
                          int64_t /*num_computed_tokens*/) override {
    return std::get_if<RecurrentRequestState>(&req) == nullptr ? WrongKind() : OkStatus();
  }

  Status Release(RequestGroupState& req) override {
    auto* s = std::get_if<RecurrentRequestState>(&req);
    if (s == nullptr) return WrongKind();
    if (s->slot >= 0) {
      const int64_t slot = s->slot;
      s->slot = -1;
      return state_pool_->ReleaseSlot(slot);
    }
    return OkStatus();
  }

  Status Fork(const RequestGroupState& parent, int64_t num_children,
              std::vector<RequestGroupState>* children,
              std::vector<SlotCopy>* slot_copies) override {
    const auto* p = std::get_if<RecurrentRequestState>(&parent);
    if (p == nullptr) return WrongKind();
    if (children == nullptr || slot_copies == nullptr || num_children <= 0) {
      return InvalidArgumentError("group ", group_.group_index, ": bad fork arguments");
    }
    if (p->slot < 0) {
      return FailedPreconditionError("group ", group_.group_index,
                                     ": cannot fork a request that holds no state slot");
    }

    std::vector<int64_t> slots;
    slots.reserve(static_cast<size_t>(num_children));
    for (int64_t c = 0; c < num_children; ++c) {
      StatusOr<int64_t> slot = state_pool_->AllocateSlot();
      if (!slot.ok()) {
        for (const int64_t taken : slots) static_cast<void>(state_pool_->ReleaseSlot(taken));
        return slot.status();
      }
      slots.push_back(*slot);
    }

    children->clear();
    for (const int64_t slot : slots) {
      RecurrentRequestState child;
      child.slot = slot;
      children->emplace_back(child);
      slot_copies->push_back(SlotCopy{p->slot, slot});  // the runner copies the state
    }
    return OkStatus();
  }

 private:
  Status WrongKind() const {
    return InvalidArgumentError("request state does not match recurrent group ",
                                group_.group_index);
  }

  const CacheGroupSpec group_;
  RecurrentStatePool* const state_pool_;
};

}  // namespace

// ===========================================================================
// Factory
// ===========================================================================

StatusOr<CacheGroupManagerPtr> CreateCacheGroupManager(const CacheGroupSpec& group,
                                                       KvBlockPool* pool,
                                                       RecurrentStatePool* state_pool) {
  if (group.layers.empty()) {
    return InvalidArgumentError("group ", group.group_index, " has no layers");
  }

  if (std::holds_alternative<RecurrentRetention>(group.spec.retention)) {
    if (state_pool == nullptr) {
      return InvalidArgumentError("recurrent group ", group.group_index, " needs a state pool");
    }
    return CacheGroupManagerPtr(new RecurrentStateManager(group, state_pool));
  }

  if (pool == nullptr) {
    return InvalidArgumentError("token-wise group ", group.group_index, " needs a block pool");
  }
  INFERX_RETURN_IF_ERROR(group.spec.Validate(pool->BlockSize()));

  const Retention& retention = group.spec.retention;
  if (std::holds_alternative<FullRetention>(retention)) {
    return CacheGroupManagerPtr(new FullAttentionManager(group, pool));
  }
  if (std::holds_alternative<WindowRetention>(retention)) {
    return CacheGroupManagerPtr(new SlidingWindowManager(group, pool));
  }
  if (std::holds_alternative<ChunkRetention>(retention)) {
    return CacheGroupManagerPtr(new ChunkedLocalManager(group, pool));
  }
  return CacheGroupManagerPtr(new CrossAttentionManager(group, pool));  // EncoderRetention
}

}  // namespace inferx