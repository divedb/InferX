#include "inferx/cache/cache_manager.h"

#include <algorithm>
#include <cstring>
#include <utility>

#include "inferx/cache/sha256.h"

namespace inferx {
namespace {

/// Domain separator for the facade's chain links, versioned so an encoding
/// change cannot silently mix with chains computed the old way.
constexpr char kChainDomain[] = "inferx.kv.chainhash.v1";

}  // namespace

BlockHash ChainHash(const BlockHash& parent, absl::Span<const TokenId> tokens,
                    absl::Span<const uint8_t> salt) {
  Sha256 sha;
  sha.Update(kChainDomain, std::strlen(kChainDomain));
  sha.Update(parent.bytes.data(), parent.bytes.size());

  const auto append_u64 = [&sha](uint64_t value) {
    for (int i = 0; i < 8; ++i) {
      const uint8_t byte = static_cast<uint8_t>(value >> (8 * i));
      sha.Update(&byte, 1);
    }
  };
  // Length framing keeps variable-size fields unambiguous.
  append_u64(static_cast<uint64_t>(tokens.size()));
  for (const TokenId token : tokens) append_u64(static_cast<uint64_t>(token));
  append_u64(static_cast<uint64_t>(salt.size()));
  sha.Update(salt.data(), salt.size());

  BlockHash hash;
  hash.bytes = sha.Finish();
  return hash;
}

namespace {

/// Fresh per-group state for a request, in plan order.
std::vector<RequestGroupState> MakeGroupStates(
    const std::vector<CacheGroupManagerPtr>& managers) {
  std::vector<RequestGroupState> groups;
  groups.reserve(managers.size());
  for (const CacheGroupManagerPtr& manager : managers) {
    if (manager->IsTokenWise()) {
      groups.emplace_back(TokenWiseRequestState{});
    } else {
      groups.emplace_back(RecurrentRequestState{});
    }
  }
  return groups;
}

/// What a Reserve would consume, split by the pool that pays for it.
struct Need {
  int64_t blocks = 0;  ///< From the block pool, over the token-wise groups.
  int64_t slots = 0;   ///< From the state pool, over the recurrent group.
};

Need ComputeNeed(const std::vector<CacheGroupManagerPtr>& managers,
                 const std::vector<RequestGroupState>& groups, const GrowthTarget& target) {
  Need need;
  for (size_t g = 0; g < managers.size(); ++g) {
    const int64_t n = managers[g]->NumNewBlocksNeeded(groups[g], target);
    (managers[g]->IsTokenWise() ? need.blocks : need.slots) += n;
  }
  return need;
}

Status CheckTarget(const GrowthTarget& target) {
  if (target.num_computed_tokens < 0 || target.target_tokens < target.num_computed_tokens) {
    return InvalidArgumentError("bad growth target: computed=", target.num_computed_tokens,
                                " target=", target.target_tokens);
  }
  return OkStatus();
}

}  // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

CacheManager::CacheManager(KvBlockPool* pool, RecurrentStatePool* state_pool, CachePlan plan,
                           bool enable_prefix_caching)
    : pool_(pool),
      state_pool_(state_pool),
      plan_(std::move(plan)),
      enable_prefix_caching_(enable_prefix_caching) {}

StatusOr<std::unique_ptr<CacheManager>> CacheManager::Create(KvBlockPool* pool,
                                                             RecurrentStatePool* state_pool,
                                                             const CachePlan& plan,
                                                             bool enable_prefix_caching) {
  if (plan.groups.empty()) return InvalidArgumentError("cache plan has no groups");
  if (plan.NumTokenWiseGroups() > 0) {
    if (pool == nullptr)
      return InvalidArgumentError("the plan has token-wise groups but no block pool");
    if (plan.block_size != pool->BlockSize()) {
      return InvalidArgumentError("plan block size ", plan.block_size,
                                  " differs from the pool's ", pool->BlockSize());
    }
  }
  if (plan.RecurrentGroup() != nullptr && state_pool == nullptr) {
    return InvalidArgumentError("the plan has a recurrent group but no state pool");
  }

  std::unique_ptr<CacheManager> manager(
      new CacheManager(pool, state_pool, plan, enable_prefix_caching));
  for (const CacheGroupSpec& group : manager->plan_.groups) {
    INFERX_ASSIGN_OR_RETURN(auto group_manager,
                            CreateCacheGroupManager(group, pool, state_pool));
    manager->managers_.push_back(std::move(group_manager));
  }
  return manager;
}

CacheManager::~CacheManager() {
  // Return everything to the pools, which are expected to outlive us.
  for (auto& entry : requests_) static_cast<void>(ReleaseRequest(entry.second));
}

// ---------------------------------------------------------------------------
// Hashing and hit search
// ---------------------------------------------------------------------------

void CacheManager::ExtendHashes(RequestState& request, absl::Span<const TokenId> prompt,
                                absl::Span<const TokenId> output, int64_t num_blocks) const {
  const int64_t bs = plan_.block_size;
  const int64_t total = static_cast<int64_t>(prompt.size() + output.size());
  std::vector<TokenId> block_tokens(static_cast<size_t>(bs));

  for (int64_t b = static_cast<int64_t>(request.hashes.size()); b < num_blocks; ++b) {
    if ((b + 1) * bs > total) break;  // not a full block yet
    for (int64_t j = 0; j < bs; ++j) {
      const size_t pos = static_cast<size_t>(b * bs + j);
      block_tokens[static_cast<size_t>(j)] =
          pos < prompt.size() ? prompt[pos] : output[pos - prompt.size()];
    }
    // The chain is rooted in a zero parent; the salt is mixed into the first
    // block only, so it namespaces every later digest through the chain.
    const BlockHash parent = b == 0 ? BlockHash{} : request.hashes[static_cast<size_t>(b) - 1];
    const absl::Span<const uint8_t> extra =
        b == 0 ? absl::MakeConstSpan(request.salt) : absl::Span<const uint8_t>{};
    request.hashes.push_back(ChainHash(parent, absl::MakeConstSpan(block_tokens), extra));
  }
}

CacheHit CacheManager::FindAgreedHit(absl::Span<const BlockHash> hashes, int64_t max_tokens,
                                     std::vector<CacheHit>* per_group) const {
  const int64_t bs = plan_.block_size;
  const size_t token_wise = static_cast<size_t>(plan_.NumTokenWiseGroups());
  per_group->assign(token_wise, CacheHit{});

  // Each group returns its longest hit not above the cap. If they differ, the
  // agreed length is the shortest, and every group must search again under
  // that cap: a window group's hit at a shorter length is a different set of
  // blocks, and may not exist. The cap strictly shrinks, so this terminates.
  int64_t cap = max_tokens;
  while (cap >= bs) {
    int64_t agreed = cap;
    for (size_t g = 0; g < token_wise; ++g) {
      (*per_group)[g] = managers_[g]->FindLongestCacheHit(hashes, cap);
      agreed = std::min(agreed, (*per_group)[g].num_tokens);
    }
    if (agreed == cap) {
      CacheHit hit;
      hit.num_tokens = cap;
      return hit;
    }
    cap = agreed;
  }
  per_group->assign(token_wise, CacheHit{});
  return CacheHit{};
}

// ---------------------------------------------------------------------------
// Admission and growth
// ---------------------------------------------------------------------------

StatusOr<CacheManager::PrefixAdoption> CacheManager::AdoptPrefix(
    uint64_t request_id, absl::Span<const TokenId> prompt, std::string_view cache_salt,
    int64_t max_cached_tokens) {
  if (requests_.contains(request_id)) {
    return InvalidArgumentError("request ", request_id, " is already registered");
  }

  RequestState request;
  request.groups = MakeGroupStates(managers_);
  request.salt.assign(cache_salt.begin(), cache_salt.end());

  PrefixAdoption adoption;
  if (enable_prefix_caching_) {
    ++stats_.prefix_queries;
    stats_.prefix_queried_tokens += static_cast<int64_t>(prompt.size());

    // KV blocks cannot be reused without the matching recurrent state, and
    // state boundary sharing does not exist yet, so hybrid models never hit.
    const int64_t bs = plan_.block_size;
    const int64_t cap = std::min<int64_t>(static_cast<int64_t>(prompt.size()),
                                          std::max<int64_t>(0, max_cached_tokens));
    const int64_t num_blocks = cap / bs;
    if (!HasRecurrentGroup() && num_blocks > 0) {
      ExtendHashes(request, prompt, {}, num_blocks);

      std::vector<CacheHit> per_group;
      const CacheHit agreed = FindAgreedHit(request.hashes, num_blocks * bs, &per_group);

      if (agreed.num_tokens > 0) {
        const size_t token_wise = static_cast<size_t>(NumTokenWiseGroups());
        for (size_t g = 0; g < token_wise; ++g) {
          const Status status = managers_[g]->AdoptCached(request.groups[g], per_group[g]);
          if (!status.ok()) {
            for (size_t h = 0; h < g; ++h) {
              static_cast<void>(managers_[h]->Release(request.groups[h]));
            }
            return status;
          }
        }
        adoption.cached_tokens = agreed.num_tokens;
        adoption.group_block_ids.reserve(token_wise);
        for (CacheHit& hit : per_group)
          adoption.group_block_ids.push_back(std::move(hit.blocks));

        ++stats_.prefix_hits;
        stats_.prefix_hit_tokens += agreed.num_tokens;
      }
    }
  }

  requests_.emplace(request_id, std::move(request));
  return adoption;
}

Status CacheManager::Reserve(uint64_t request_id, const GrowthTarget& target) {
  INFERX_RETURN_IF_ERROR(CheckTarget(target));

  auto it = requests_.find(request_id);
  bool created = false;
  if (it == requests_.end()) {
    RequestState request;
    request.groups = MakeGroupStates(managers_);
    it = requests_.emplace(request_id, std::move(request)).first;
    created = true;
  }
  RequestState& request = it->second;
  const auto fail = [&](Status status) {
    if (created) requests_.erase(it);  // do not leave a registration behind
    return status;
  };

  const size_t num_groups = managers_.size();

  // 1. Roll off tokens below each group's retention floor. These blocks are
  //    dead for the request whatever happens next.
  for (size_t g = 0; g < num_groups; ++g) {
    if (!managers_[g]->IsTokenWise()) continue;
    const Status status =
        managers_[g]->ReleaseSkipped(request.groups[g], target.num_computed_tokens);
    if (!status.ok()) return fail(status);
  }

  // 2. Check capacity across every group before touching any of them.
  const Need need = ComputeNeed(managers_, request.groups, target);
  const int64_t free_blocks = NumFreeBlocks();
  const int64_t free_slots = NumFreeSlots();
  if (need.blocks > free_blocks || need.slots > free_slots) {
    return fail(ResourceExhaustedError(
        "cannot reserve for request ", request_id, ": needs ", need.blocks, " blocks (",
        free_blocks, " available) and ", need.slots, " state slots (", free_slots, " free)"));
  }

  // 3. Reserve each group; undo the ones already done if a later one fails.
  std::vector<Granted> granted(num_groups);
  for (size_t g = 0; g < num_groups; ++g) {
    StatusOr<Granted> result = managers_[g]->Reserve(request.groups[g], target);
    if (!result.ok()) {
      for (size_t h = g; h-- > 0;) {
        static_cast<void>(managers_[h]->Rollback(request.groups[h], granted[h]));
      }
      return fail(result.status());
    }
    granted[g] = std::move(*result);
  }

  // 4. Only now queue the device work, so a failed call leaves none behind.
  for (const Granted& grant : granted) {
    if (grant.slot.has_value()) pending_work_.slots_to_reset.push_back(*grant.slot);
    pending_work_.block_copies.insert(pending_work_.block_copies.end(), grant.copies.begin(),
                                      grant.copies.end());
  }
  return OkStatus();
}

bool CacheManager::CanReserve(uint64_t request_id, const GrowthTarget& target) const {
  if (!CheckTarget(target).ok()) return false;

  const auto it = requests_.find(request_id);
  // Conservative: blocks a roll-off would free are not counted.
  const Need need = it != requests_.end()
                        ? ComputeNeed(managers_, it->second.groups, target)
                        : ComputeNeed(managers_, MakeGroupStates(managers_), target);
  return need.blocks <= NumFreeBlocks() && need.slots <= NumFreeSlots();
}

Status CacheManager::PublishComputed(uint64_t request_id, absl::Span<const TokenId> prompt,
                                     absl::Span<const TokenId> output,
                                     int64_t num_computed_tokens) {
  // Entries for a hybrid model could never be reused (see AdoptPrefix).
  if (!enable_prefix_caching_ || HasRecurrentGroup()) return OkStatus();

  const auto it = requests_.find(request_id);
  if (it == requests_.end()) {
    return InvalidArgumentError("request ", request_id, " is not registered");
  }
  RequestState& request = it->second;

  const int64_t total = static_cast<int64_t>(prompt.size() + output.size());
  const int64_t computed = std::max<int64_t>(0, std::min(num_computed_tokens, total));
  ExtendHashes(request, prompt, output, computed / plan_.block_size);

  const size_t token_wise = static_cast<size_t>(NumTokenWiseGroups());
  for (size_t g = 0; g < token_wise; ++g) {
    INFERX_RETURN_IF_ERROR(
        managers_[g]->PublishCompleted(request.groups[g], request.hashes, computed));
  }
  return OkStatus();
}

Status CacheManager::Fork(uint64_t parent_request_id,
                          absl::Span<const uint64_t> child_request_ids) {
  const auto parent_it = requests_.find(parent_request_id);
  if (parent_it == requests_.end()) {
    return InvalidArgumentError("parent request ", parent_request_id, " is not registered");
  }
  if (child_request_ids.empty()) return InvalidArgumentError("fork needs at least one child");
  for (size_t i = 0; i < child_request_ids.size(); ++i) {
    if (requests_.contains(child_request_ids[i]) ||
        std::find(child_request_ids.begin(),
                  child_request_ids.begin() + static_cast<std::ptrdiff_t>(i),
                  child_request_ids[i]) !=
            child_request_ids.begin() + static_cast<std::ptrdiff_t>(i)) {
      return InvalidArgumentError("child request id ", child_request_ids[i],
                                  " is already in use");
    }
  }

  // Copy what children inherit now: emplacing them below may rehash the map
  // and invalidate `parent_it`.
  const RequestState& parent = parent_it->second;
  const std::vector<BlockHash> hashes = parent.hashes;
  const std::vector<uint8_t> salt = parent.salt;

  const size_t num_groups = managers_.size();
  const int64_t num_children = static_cast<int64_t>(child_request_ids.size());
  std::vector<std::vector<RequestGroupState>> forked(num_groups);
  std::vector<SlotCopy> slot_copies;

  for (size_t g = 0; g < num_groups; ++g) {
    const Status status =
        managers_[g]->Fork(parent.groups[g], num_children, &forked[g], &slot_copies);
    if (!status.ok()) {
      // Give back the children already made in earlier groups.
      for (size_t h = 0; h < g; ++h) {
        for (RequestGroupState& child : forked[h]) {
          static_cast<void>(managers_[h]->Release(child));
        }
      }
      return status;
    }
  }

  for (size_t c = 0; c < child_request_ids.size(); ++c) {
    RequestState child;
    child.hashes = hashes;
    child.salt = salt;
    child.groups.reserve(num_groups);
    for (size_t g = 0; g < num_groups; ++g) child.groups.push_back(std::move(forked[g][c]));
    requests_.emplace(child_request_ids[c], std::move(child));
  }
  pending_work_.slot_copies.insert(pending_work_.slot_copies.end(), slot_copies.begin(),
                                   slot_copies.end());
  return OkStatus();
}

Status CacheManager::ReleaseRequest(RequestState& request) {
  Status first_error = OkStatus();
  for (size_t g = 0; g < managers_.size() && g < request.groups.size(); ++g) {
    const Status status = managers_[g]->Release(request.groups[g]);
    if (!status.ok() && first_error.ok()) first_error = status;  // keep releasing the rest
  }
  return first_error;
}

Status CacheManager::Release(uint64_t request_id) {
  const auto it = requests_.find(request_id);
  if (it == requests_.end()) return OkStatus();
  const Status status = ReleaseRequest(it->second);
  requests_.erase(it);
  return status;
}

CacheManager::RunnerWork CacheManager::TakeRunnerWork() {
  RunnerWork work = std::move(pending_work_);
  pending_work_ = RunnerWork{};
  return work;
}

Status CacheManager::ResetPrefixCache() {
  return pool_ != nullptr ? pool_->ResetPrefixCache() : OkStatus();
}

// ---------------------------------------------------------------------------
// Views and counters
// ---------------------------------------------------------------------------

int64_t CacheManager::NumTokenWiseGroups() const { return plan_.NumTokenWiseGroups(); }

bool CacheManager::HasRecurrentGroup() const { return plan_.RecurrentGroup() != nullptr; }

absl::Span<const BlockId> CacheManager::GroupBlocks(uint64_t request_id,
                                                    int64_t group_index) const {
  if (group_index < 0 || group_index >= NumTokenWiseGroups()) return {};
  const auto it = requests_.find(request_id);
  if (it == requests_.end()) return {};
  const auto* state =
      std::get_if<TokenWiseRequestState>(&it->second.groups[static_cast<size_t>(group_index)]);
  return state != nullptr ? absl::MakeConstSpan(state->blocks) : absl::Span<const BlockId>{};
}

int64_t CacheManager::GroupSkippedTokens(int64_t group_index,
                                         int64_t num_computed_tokens) const {
  if (group_index < 0 || group_index >= NumTokenWiseGroups()) return 0;
  return managers_[static_cast<size_t>(group_index)]->NumSkippedTokens(num_computed_tokens);
}

int64_t CacheManager::StateSlot(uint64_t request_id) const {
  if (!HasRecurrentGroup()) return -1;
  const auto it = requests_.find(request_id);
  if (it == requests_.end()) return -1;
  const auto* state = std::get_if<RecurrentRequestState>(&it->second.groups.back());
  return state != nullptr ? state->slot : -1;
}

bool CacheManager::PrefixCachingEnabled() const { return enable_prefix_caching_; }
int64_t CacheManager::BlockSize() const { return plan_.block_size; }
int64_t CacheManager::NumFreeBlocks() const {
  return pool_ != nullptr ? pool_->NumFreeBlocks() : 0;
}
int64_t CacheManager::NumBlocks() const { return pool_ != nullptr ? pool_->NumBlocks() : 0; }
int64_t CacheManager::NumFreeSlots() const {
  return state_pool_ != nullptr ? state_pool_->NumFreeSlots() : 0;
}
double CacheManager::Usage() const { return pool_ != nullptr ? pool_->Usage() : 0.0; }
int64_t CacheManager::NumCachedBlocks() const {
  return pool_ != nullptr ? pool_->NumCachedBlocks() : 0;
}

}  // namespace inferx