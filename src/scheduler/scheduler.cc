#include "inferx/scheduler/scheduler.h"

#include <algorithm>
#include <utility>

#include "inferx/scheduler/prefix_cache.h"

namespace inferx::scheduler {
namespace {

/// One admitted sequence: its tokens, its blocks, and where it is.
struct Sequence {
  RequestId id = 0;
  SamplingParams params;

  /// Prompt followed by everything generated. Positions are indices into this.
  std::vector<int32_t> tokens;
  int64_t prompt_len = 0;

  /// Tokens whose K/V is already in the cache. Everything below this has been
  /// through a forward pass; everything at or above it has not.
  int64_t cached = 0;

  /// Leading tokens whose KV came from the prefix cache rather than from a
  /// forward pass this sequence ran (§6.3). Those blocks are the tree's, not
  /// this sequence's, and freeing them here would pull them out from under
  /// everyone else reading the same prefix.
  int64_t acquired = 0;

  std::unique_ptr<BlockTable> table;

  FinishReason finish = FinishReason::kNotFinished;

  int64_t generated() const {
    return static_cast<int64_t>(tokens.size()) - prompt_len;
  }

  /// Tokens waiting to be run: what is left of the prompt while prefilling,
  /// one token once it is generating.
  int64_t pending() const {
    return static_cast<int64_t>(tokens.size()) - cached;
  }
};

/// What one sequence contributed to the batch that was just prepared.
///
/// `CommitStep` needs both numbers and they are no longer the same. Chunked
/// prefill separates "how much of this sequence ran" from "did it produce a
/// token": a middle chunk moves `cached` forward by a thousand tokens and
/// samples nothing, and the step before a prompt's last chunk lands looks
/// identical to one that never ran it.
struct Contribution {
  /// Index into `running`.
  int64_t seq = 0;
  /// Tokens this sequence put in the batch: a chunk while prefilling, 1 while
  /// decoding.
  int64_t tokens = 0;
  /// Whether it has a row in `logits_indices`, and so a token coming back.
  bool sampled = false;
};

}  // namespace

const char* FinishReasonName(FinishReason reason) {
  switch (reason) {
    case FinishReason::kNotFinished:
      return "not_finished";
    case FinishReason::kStopToken:
      return "stop_token";
    case FinishReason::kMaxTokens:
      return "max_tokens";
    case FinishReason::kCancelled:
      return "cancelled";
    case FinishReason::kOutOfMemory:
      return "out_of_memory";
    case FinishReason::kContextLimit:
      return "context_limit";
  }
  return "?";
}

struct Scheduler::Impl {
  SchedulerConfig config;
  KvBlockPool* pool = nullptr;

  /// §6.3. Constructed always, consulted only when the config enables it, so
  /// that the disabled path is one branch rather than a null check everywhere.
  std::unique_ptr<PrefixCache> cache;

  /// Hands a finished or evicted sequence's blocks to whoever should have them.
  ///
  /// With the cache on, the complete ones are offered to the tree and the rest
  /// go back to the pool; with it off, all of them go back. Every place a
  /// sequence loses its blocks goes through here, because getting the split
  /// wrong in one of them either leaks or, worse, frees a block another
  /// sequence is still reading through the tree.
  void ReleaseBlocks(Sequence* seq) {
    if (seq->table == nullptr) return;

    if (config.enable_prefix_cache) {
      cache->Finish(seq->tokens, seq->cached, seq->table->blocks(),
                    seq->acquired);
    } else if (!seq->table->blocks().empty()) {
      (void)pool->FreeBlocks(seq->table->blocks());
    }

    seq->table->Clear();
    seq->acquired = 0;
  }

  std::deque<Sequence> waiting;
  std::vector<Sequence> running;
  std::vector<Completion> completed;
  std::vector<RequestId> admitted;

  /// What the last prepared batch asked of each sequence, in the order the
  /// batch lists them, so `CommitStep` can put the results back where they
  /// belong. Rebuilt every step.
  std::vector<Contribution> step;

  /// Tokens each running sequence contributes to the step being planned.
  /// Scratch, kept across steps only to avoid reallocating it.
  std::vector<int64_t> take;

  int64_t max_blocks_per_seq = 0;

  /// Cumulative count of §8.2 preemptions, for `Scheduler::preemptions()`.
  int64_t preemptions = 0;

  /// Gives back everything still held.
  ///
  /// `BlockTable` carries no pointer to the pool it drew from, so it cannot
  /// return its own blocks and the scheduler has to do it. Without this a
  /// scheduler destroyed with requests in flight -- a server shutting down
  /// mid-stream, a test that stops once it has seen what it came for -- leaves
  /// those blocks marked used in a pool that outlives it, and nothing ever
  /// hands them back.
  ~Impl() {
    for (Sequence& seq : running) ReleaseBlocks(&seq);
    for (Sequence& seq : waiting) ReleaseBlocks(&seq);

    // Destroyed before the pool outlives us: the tree owns blocks too, and its
    // destructor is what returns them.
    cache.reset();
  }

  /// Grows a sequence's block table to cover `tokens` tokens.
  ///
  /// Reports *which* way it failed rather than returning a Status, because the
  /// two are not the same event and the caller does opposite things with them.
  /// A full context is the sequence's own fault and terminal -- no amount of
  /// freeing helps, because the limit is on where its tokens may live rather
  /// than on whether room exists. An empty pool is a shortage, which §8.2 says
  /// to resolve by taking the memory from someone else.
  ///
  /// They used to share one ResourceExhausted, which is how a single sequence
  /// reaching `max_seq_len` could take down every request in flight: the error
  /// left `PrepareStep`, and the engine's only answer to that is to fail the
  /// whole batch.
  enum class Grow {
    kOk,
    /// `max_seq_len` reached. Terminal for this sequence.
    kContextFull,
    /// The pool has no free block. Someone has to give one up.
    kPoolEmpty,
  };

  Grow Reserve(Sequence* seq, int64_t tokens) {
    while (seq->table->capacity_tokens() < tokens) {
      if (seq->table->size() >= max_blocks_per_seq) return Grow::kContextFull;

      StatusOr<int32_t> block = pool->AllocateBlock();

      // An unreferenced cached block is free memory that happens to still hold
      // something useful, so the pool being empty is not the same as there
      // being nothing left. §6.3 puts eviction here, on allocation failure,
      // rather than on a timer: the cache should be as large as it can be right
      // up until the moment the memory is actually wanted.
      if (!block.ok() && config.enable_prefix_cache && cache->Evict(1) > 0) {
        block = pool->AllocateBlock();
      }

      if (!block.ok()) return Grow::kPoolEmpty;

      seq->table->Append(*block);
    }
    return Grow::kOk;
  }

  /// Sends a running sequence back to the queue and returns its KV to the pool.
  ///
  /// **Recompute, not swap** (§8.2). The blocks go back immediately and the
  /// sequence re-reads its own history later, which costs a prefill it has
  /// already paid for once and costs nothing on the PCIe bus. Swapping would
  /// preserve the work at the price of a D2H copy the overlap pipeline would
  /// then have to hide, and with a prefix cache in front of it (M5's remaining
  /// item) recompute is nearly always the better trade -- that is the bet §8.2
  /// makes, and it is worth measuring rather than assuming.
  ///
  /// What survives is everything except the KV: `tokens` still holds the prompt
  /// *and* whatever was generated before the preemption, so the sequence
  /// resumes by prefilling all of it and predicting the next token. A client
  /// mid-stream sees a pause, not a restart.
  void Preempt(size_t index) {
    Sequence victim = std::move(running[index]);
    running.erase(running.begin() + static_cast<long>(index));

    // Before `cached` is reset, because that is what says how much of the
    // sequence has real KV behind it -- and with §6.3 on, this is a donation
    // rather than a discard. The history goes into the tree, where it stays
    // evictable but matchable, so the sequence's own re-admission is likely to
    // find it and the recompute §8.2 budgets for never happens. That is the
    // trade T7 makes against swapping, and this is where it pays.
    ReleaseBlocks(&victim);
    victim.table.reset();

    // Everything it had cached is gone, so it starts again from token zero.
    victim.cached = 0;

    // The head of the queue, not the back. It was admitted before anything
    // still waiting and a client has already seen tokens from it; letting newer
    // requests overtake it would be unfair in the one direction FCFS exists to
    // prevent, and could starve it indefinitely under sustained load.
    waiting.push_front(std::move(victim));

    ++preemptions;
  }

  void Retire(Sequence* seq, FinishReason reason) {
    Completion c;
    c.id = seq->id;
    c.reason = reason;
    c.output_tokens.assign(seq->tokens.begin() + seq->prompt_len,
                           seq->tokens.end());
    completed.push_back(std::move(c));

    // Released here rather than at the next step boundary: the blocks are dead
    // the moment the sequence is, and holding them would refuse admission to a
    // request that could have used them. With §6.3 on, "released" means the
    // complete ones are offered to the tree -- still reclaimable, but matchable
    // by the next request that shares this prompt.
    ReleaseBlocks(seq);
  }

  /// Fills `take` with how many tokens each running sequence contributes to the
  /// next step (§8.1).
  ///
  /// Separate from emission because the budget is handed out in a different
  /// order than the batch is written in: emission has to run in `running` order
  /// -- the attention plan requires rows grouped by sequence and in ascending
  /// sequence index -- while the budget goes to decodes first regardless of
  /// where they sit in that order.
  void Plan() {
    const int64_t num_seqs = static_cast<int64_t>(running.size());

    take.assign(static_cast<size_t>(num_seqs), 0);
    int64_t budget = config.max_batch_tokens;

    // Decodes first. One token each, served ahead of any prompt so that a
    // decoding sequence's inter-token time does not include some other
    // request's prefill. Bounding that is the entire reason chunking exists, so
    // it is the priority rule and not a tuning choice: measured at 10.6x on p99
    // TBT in bench/tbt_bench.cc.
    //
    // A prompt's final chunk has one token pending too, and lands here. That is
    // right -- it is one token of work either way, and it is the step that
    // request has been waiting through its whole prefill for.
    for (int64_t s = 0; s < num_seqs && budget > 0; ++s) {
      if (running[static_cast<size_t>(s)].pending() == 1) {
        take[static_cast<size_t>(s)] = 1;
        --budget;
      }
    }

    // Then prompts fill what is left, FCFS so the oldest finishes first (§8.3).
    // A prompt that does not fit is *chunked* rather than deferred: it takes
    // the rest of the budget now and resumes next step. The alternative --
    // waiting for a step it fits in whole -- is what made a long prompt able to
    // stall behind a busy engine indefinitely, and what made a prompt longer
    // than the budget impossible to serve at all.
    for (int64_t s = 0; s < num_seqs && budget > 0; ++s) {
      const int64_t pending = running[static_cast<size_t>(s)].pending();
      if (pending <= 1) continue;

      const int64_t chunk = std::min(pending, budget);
      take[static_cast<size_t>(s)] = chunk;
      budget -= chunk;
    }
  }

  /// Moves waiting requests into `running` while there is room for them.
  void Admit() {
    while (!waiting.empty() &&
           static_cast<int64_t>(running.size()) < config.max_running) {
      Sequence seq = std::move(waiting.front());

      // Cancelled while queued: retire without touching the pool.
      if (seq.finish == FinishReason::kCancelled) {
        waiting.pop_front();
        Retire(&seq, FinishReason::kCancelled);
        continue;
      }

      const int64_t needed = static_cast<int64_t>(seq.tokens.size());

      if (needed > config.max_seq_len) {
        waiting.pop_front();
        seq.table = std::make_unique<BlockTable>(pool->block_size());
        Retire(&seq, FinishReason::kOutOfMemory);
        continue;
      }

      seq.table = std::make_unique<BlockTable>(pool->block_size());

      // §6.3, step 4 of §4: the prefix match happens at admission, because it
      // decides how much of the prompt there is left to reserve *and* how much
      // there is left to run. A hit here is a prefill that never happens.
      //
      // Capped one token short of the sequence, since a request whose every
      // token is already cached would have nothing to forward and so no logits
      // to sample its first token from.
      if (config.enable_prefix_cache) {
        const PrefixCache::Match match = cache->Acquire(seq.tokens, needed - 1);

        for (const int32_t block : match.blocks) seq.table->Append(block);

        // Those tokens are already through a forward pass -- someone else's,
        // but the KV is the same, because a key depends on its prefix and the
        // path through the tree is that prefix.
        seq.acquired = match.tokens;
        seq.cached = match.tokens;
      }

      // Reserved for the prompt only, and only the part of it not already in
      // the cache. Growth for generated tokens happens per step, which is what
      // makes the pool shared rather than partitioned.
      //
      // Deliberately *not* a preemption site. A shortage here means there is
      // not room for a request that has not started yet, and the answer to that
      // is to leave it queued -- taking KV from a sequence that is already
      // producing tokens, to admit one that is not, would trade real progress
      // for none. §8.2's preemption exists for the other case: a sequence that
      // is already running and can no longer grow.
      const Grow reserved = Reserve(&seq, needed);

      if (reserved != Grow::kOk) {
        // Not enough blocks right now. Give back what was taken -- including
        // the reference on the matched prefix, which would otherwise pin it
        // against eviction on behalf of a request that is not even running --
        // and leave the request queued. A running sequence finishing will free
        // room, and FCFS means this one goes first when it does.
        ReleaseBlocks(&seq);
        seq.cached = 0;
        waiting.front() = std::move(seq);
        return;
      }

      // Counted here rather than at the lookup: this is the point at which
      // the match is actually going to be used.
      if (config.enable_prefix_cache) {
        cache->RecordAdmission(seq.acquired, needed);
      }

      waiting.pop_front();
      admitted.push_back(seq.id);
      running.push_back(std::move(seq));
    }
  }
};

StatusOr<Scheduler> Scheduler::Create(const SchedulerConfig& config,
                                      KvBlockPool* pool) {
  if (pool == nullptr) return InvalidArgumentError("scheduler needs a KV pool");

  if (config.max_running <= 0 || config.max_batch_tokens <= 0 ||
      config.max_seq_len <= 0) {
    return InvalidArgumentError("scheduler config has a non-positive bound");
  }

  auto impl = std::make_unique<Impl>();
  impl->config = config;
  impl->pool = pool;
  impl->cache = std::make_unique<PrefixCache>(pool, pool->block_size());
  impl->max_blocks_per_seq =
      (config.max_seq_len + pool->block_size() - 1) / pool->block_size();

  return Scheduler(std::move(impl));
}

Scheduler::Scheduler(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
Scheduler::~Scheduler() = default;
Scheduler::Scheduler(Scheduler&&) noexcept = default;
Scheduler& Scheduler::operator=(Scheduler&&) noexcept = default;

Status Scheduler::AddRequest(RequestId id, std::vector<int32_t> prompt,
                             SamplingParams params) {
  if (prompt.empty()) return InvalidArgumentError("prompt is empty");

  if (params.max_tokens <= 0) {
    return InvalidArgumentError("max_tokens must be positive, got ",
                                params.max_tokens);
  }

  Sequence seq;
  seq.id = id;
  seq.params = std::move(params);
  seq.prompt_len = static_cast<int64_t>(prompt.size());
  seq.tokens = std::move(prompt);

  impl_->waiting.push_back(std::move(seq));

  return OkStatus();
}

void Scheduler::Cancel(RequestId id) {
  for (Sequence& seq : impl_->waiting) {
    if (seq.id == id) seq.finish = FinishReason::kCancelled;
  }
  for (Sequence& seq : impl_->running) {
    if (seq.id == id) seq.finish = FinishReason::kCancelled;
  }
}

Status Scheduler::PrepareStep(model::ForwardBatch* out_batch) {
  *out_batch = model::ForwardBatch{};
  impl_->step.clear();
  impl_->admitted.clear();

  // Cancellations take effect before admission, so a cancelled running
  // sequence releases its blocks in time for a waiting one to use them.
  for (auto it = impl_->running.begin(); it != impl_->running.end();) {
    if (it->finish == FinishReason::kCancelled) {
      impl_->Retire(&*it, FinishReason::kCancelled);
      it = impl_->running.erase(it);
    } else {
      ++it;
    }
  }

  impl_->Admit();

  // ---- Plan, and make room for the plan ----
  //
  // Reserving for the whole batch before emitting any of it is what makes
  // preemption possible at all. Preempting removes a sequence from `running`,
  // which renumbers every row after it and changes how the budget divides, so a
  // half-written batch would have to be unwound. Resolved here, the shortage
  // costs nothing but a recount: the batch is still only a vector of token
  // counts.
  //
  // Each pass removes at most one sequence from `running` -- retired or
  // preempted -- so the loop cannot run more times than there were sequences.
  const size_t initial_running = impl_->running.size();

  for (size_t attempt = 0; attempt <= initial_running; ++attempt) {
    if (impl_->running.empty()) return OkStatus();

    impl_->Plan();

    bool replan = false;

    for (size_t s = 0; s < impl_->running.size(); ++s) {
      if (impl_->take[s] == 0) continue;

      Sequence& seq = impl_->running[s];

      const Impl::Grow grew = impl_->Reserve(&seq, seq.cached + impl_->take[s]);

      if (grew == Impl::Grow::kOk) continue;

      if (grew == Impl::Grow::kContextFull) {
        // Terminal and nobody else's problem. The sequence has filled
        // max_seq_len, so there is nowhere to put its next token no matter how
        // much of the pool is free -- preempting on its behalf would free
        // memory it cannot address.
        impl_->Retire(&seq, FinishReason::kContextLimit);
        impl_->running.erase(impl_->running.begin() + static_cast<long>(s));
      } else {
        // §8.2. Take the KV from the most recently admitted sequence: it is the
        // one with the least invested, and preempting from the back is what
        // keeps FCFS admission from being undone at the other end.
        size_t victim = impl_->running.size() - 1;

        if (victim == s) {
          if (impl_->running.size() == 1) {
            // Nothing to take. It is the only sequence running, so it already
            // holds every block there is and the pool cannot be relieved.
            impl_->Retire(&seq, FinishReason::kOutOfMemory);
            impl_->running.erase(impl_->running.begin() + static_cast<long>(s));
            replan = true;
            break;
          }

          // It is itself the newest, and it is the one that needs the block --
          // so the next-newest gives it up. Self-preemption would free this
          // sequence's blocks only to demand all of them back plus one.
          victim = impl_->running.size() - 2;
        }

        impl_->Preempt(victim);
      }

      replan = true;
      break;
    }

    if (!replan) break;
  }

  if (impl_->running.empty()) return OkStatus();

  const int64_t num_seqs = static_cast<int64_t>(impl_->running.size());

  out_batch->num_seqs = num_seqs;
  out_batch->max_blocks_per_seq = impl_->max_blocks_per_seq;
  out_batch->block_table.assign(
      static_cast<size_t>(num_seqs * impl_->max_blocks_per_seq), 0);
  out_batch->seq_lens.assign(static_cast<size_t>(num_seqs), 0);

  // ---- Emit ----
  for (size_t s = 0; s < impl_->running.size(); ++s) {
    Sequence& seq = impl_->running[s];

    const int64_t take = impl_->take[s];
    const bool contributes = take > 0;

    // Growth happens *before* the row is written, and that ordering is the
    // whole point. Reserve appends a block whenever this step's tokens run past
    // what the sequence already holds, and the slots below are computed from
    // the grown table -- so a row copied beforehand would describe a table that
    // no longer exists. The token's key and value would be written into the new
    // block (slots address the cache directly) while attention, which reads
    // through this row, would find a zero there and attend to block 0 instead.
    //
    // That is R8: on exactly the step that first touches a new block, the model
    // read one block of some other sequence's history. It was invisible
    // whenever the zero happened to name the right block, which is why the
    // symptom was an *alternating* answer rather than a wrong one -- a stack
    // free list hands consecutive requests their blocks in opposite order.
    //
    // The growth itself now happens in the reserve loop above, before any row
    // is written, which preserves that ordering for the whole batch at once
    // rather than per sequence. The invariant is the same one and
    // SchedulerTest.EveryBatchSlotIsCoveredByTheBatchBlockTable still guards
    // it.

    // A sequence's row in the block table, regardless of whether it contributes
    // tokens this step -- the row index is its identity for the whole batch.
    for (int64_t b = 0; b < seq.table->size(); ++b) {
      out_batch->block_table[static_cast<size_t>(
          static_cast<int64_t>(s) * impl_->max_blocks_per_seq + b)] =
          seq.table->blocks()[static_cast<size_t>(b)];
    }

    // What the sequence's KV will hold once this step has run. Stated for every
    // sequence including the ones contributing nothing, because a deferred
    // sequence's history is exactly what cannot be read back off the query
    // rows -- it has none.
    const int64_t end = seq.cached + take;
    out_batch->seq_lens[s] = static_cast<int32_t>(end);

    if (!contributes) continue;

    const int64_t start = seq.cached;

    for (int64_t i = start; i < end; ++i) {
      int32_t block = 0;
      int64_t slot = 0;
      if (!seq.table->Locate(i, &block, &slot)) {
        return InternalError("position ", i, " of request ", seq.id,
                             " has no block");
      }

      out_batch->token_ids.push_back(seq.tokens[static_cast<size_t>(i)]);
      out_batch->positions.push_back(static_cast<int32_t>(i));
      out_batch->seq_of_token.push_back(static_cast<int32_t>(s));
      out_batch->slots.push_back(
          static_cast<int32_t>(block * impl_->pool->block_size() + slot));
    }

    // Logits come from the last token of a sequence's *own* run, and only once
    // that run has reached the end of what it has to say. A middle chunk stops
    // somewhere inside the prompt, so its last token predicts a token the
    // prompt already supplies -- computing it would cost a [1, 151936] row to
    // discard, and sampling it would append a token to a prompt still being
    // read.
    const bool complete = end == static_cast<int64_t>(seq.tokens.size());

    if (complete) {
      out_batch->logits_indices.push_back(
          static_cast<int32_t>(out_batch->token_ids.size() - 1));

      // Parallel to logits_indices, so the executor never has to ask which
      // request a row belongs to -- it just reads the row's parameters.
      out_batch->temperature.push_back(seq.params.temperature);
      out_batch->top_p.push_back(seq.params.top_p);

      // Mixed into the position so a request's successive tokens draw
      // differently while staying reproducible from its seed alone.
      out_batch->seeds.push_back(
          seq.params.seed ^
          (0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(seq.tokens.size())));
    }

    impl_->step.push_back({static_cast<int64_t>(s), take, complete});
  }

  // Every sequence was already cached and none is pending. Nothing to run, and
  // that is not an error -- the caller simply has no step to take.
  if (out_batch->token_ids.empty()) *out_batch = model::ForwardBatch{};

  return OkStatus();
}

Status Scheduler::CommitStep(const std::vector<int32_t>& sampled,
                             std::vector<TokenDelta>* out_deltas) {
  if (out_deltas != nullptr) {
    out_deltas->clear();
    out_deltas->reserve(sampled.size());
  }

  size_t expected = 0;
  for (const Contribution& c : impl_->step) expected += c.sampled ? 1 : 0;

  if (sampled.size() != expected) {
    return InvalidArgumentError("got ", sampled.size(),
                                " sampled tokens but "
                                "the batch asked for ",
                                expected);
  }

  size_t next = 0;

  for (const Contribution& contribution : impl_->step) {
    Sequence& seq = impl_->running[static_cast<size_t>(contribution.seq)];

    // Everything that went into the batch is now cached, including a chunk's
    // worth of prompt whose logits were never computed. Advanced by what ran
    // rather than set to the sequence's length, which is the same thing only
    // when the whole remainder ran.
    seq.cached += contribution.tokens;

    // A middle chunk ends here. It moved the prefill forward and produced no
    // token, so there is nothing to append, nothing to stop on, and nothing to
    // stream.
    if (!contribution.sampled) continue;

    const int32_t token = sampled[next++];

    seq.tokens.push_back(token);

    const bool is_stop =
        std::find(seq.params.stop_tokens.begin(), seq.params.stop_tokens.end(),
                  token) != seq.params.stop_tokens.end();

    if (is_stop) {
      seq.finish = FinishReason::kStopToken;
    } else if (seq.generated() >= seq.params.max_tokens) {
      seq.finish = FinishReason::kMaxTokens;
    }

    // Emitted after the stop checks, so `finish` is already decided and a
    // streaming caller can close the stream on this same delta.
    if (out_deltas != nullptr) {
      out_deltas->push_back({seq.id, token, seq.finish});
    }
  }

  impl_->step.clear();

  // Retire in one pass afterwards, so indices stay valid while the loop above
  // is using them.
  for (auto it = impl_->running.begin(); it != impl_->running.end();) {
    if (it->finish != FinishReason::kNotFinished) {
      impl_->Retire(&*it, it->finish);
      it = impl_->running.erase(it);
    } else {
      ++it;
    }
  }

  return OkStatus();
}

std::vector<RequestId> Scheduler::TakeAdmitted() {
  std::vector<RequestId> result;
  result.swap(impl_->admitted);
  return result;
}

std::vector<Completion> Scheduler::TakeCompleted() {
  std::vector<Completion> out;
  out.swap(impl_->completed);
  return out;
}

bool Scheduler::HasWork() const {
  return !impl_->running.empty() || !impl_->waiting.empty();
}

int64_t Scheduler::num_running() const {
  return static_cast<int64_t>(impl_->running.size());
}

int64_t Scheduler::num_waiting() const {
  return static_cast<int64_t>(impl_->waiting.size());
}

int64_t Scheduler::blocks_in_use() const {
  // The tree's blocks are not any sequence's, and they are reclaimable on
  // demand, so counting them here would report a healthy cache as pressure.
  return impl_->pool->used_blocks() - impl_->cache->cached_blocks();
}

int64_t Scheduler::cached_blocks() const {
  return impl_->cache->cached_blocks();
}

int64_t Scheduler::prefix_hit_tokens() const {
  return impl_->cache->hit_tokens();
}

int64_t Scheduler::prefix_miss_tokens() const {
  return impl_->cache->miss_tokens();
}

int64_t Scheduler::evicted_blocks() const {
  return impl_->cache->evicted_blocks();
}

int64_t Scheduler::preemptions() const { return impl_->preemptions; }

}  // namespace inferx::scheduler
