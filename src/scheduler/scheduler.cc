#include "inferx/scheduler/scheduler.h"

#include <algorithm>
#include <utility>

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
    case FinishReason::kNotFinished: return "not_finished";
    case FinishReason::kStopToken:   return "stop_token";
    case FinishReason::kMaxTokens:   return "max_tokens";
    case FinishReason::kCancelled:   return "cancelled";
    case FinishReason::kOutOfMemory: return "out_of_memory";
  }
  return "?";
}

struct Scheduler::Impl {
  SchedulerConfig config;
  KvBlockPool* pool = nullptr;

  std::deque<Sequence> waiting;
  std::vector<Sequence> running;
  std::vector<Completion> completed;

  /// What the last prepared batch asked of each sequence, in the order the
  /// batch lists them, so `CommitStep` can put the results back where they
  /// belong. Rebuilt every step.
  std::vector<Contribution> step;

  /// Tokens each running sequence contributes to the step being planned.
  /// Scratch, kept across steps only to avoid reallocating it.
  std::vector<int64_t> take;

  int64_t max_blocks_per_seq = 0;

  /// Gives back everything still held.
  ///
  /// `BlockTable` carries no pointer to the pool it drew from, so it cannot
  /// return its own blocks and the scheduler has to do it. Without this a
  /// scheduler destroyed with requests in flight -- a server shutting down
  /// mid-stream, a test that stops once it has seen what it came for -- leaves
  /// those blocks marked used in a pool that outlives it, and nothing ever
  /// hands them back.
  ~Impl() {
    const auto release = [this](Sequence& seq) {
      if (seq.table != nullptr && !seq.table->blocks().empty()) {
        (void)pool->FreeBlocks(seq.table->blocks());
        seq.table->Clear();
      }
    };

    for (Sequence& seq : running) release(seq);
    for (Sequence& seq : waiting) release(seq);
  }

  /// Grows a sequence's block table to cover `tokens` tokens.
  ///
  /// Returns ResourceExhausted rather than throwing, because running out mid
  /// sequence is a scheduling event (§8.2) and not a fault: at M3 it retires the
  /// sequence, and at M5 it will preempt one instead.
  Status Reserve(Sequence* seq, int64_t tokens) {
    while (seq->table->capacity_tokens() < tokens) {
      if (seq->table->size() >= max_blocks_per_seq) {
        return ResourceExhaustedError("sequence would exceed max_seq_len");
      }

      INFERX_ASSIGN_OR_RETURN(const int32_t block, pool->AllocateBlock());
      seq->table->Append(block);
    }
    return OkStatus();
  }

  void Retire(Sequence* seq, FinishReason reason) {
    Completion c;
    c.id = seq->id;
    c.reason = reason;
    c.output_tokens.assign(seq->tokens.begin() + seq->prompt_len,
                           seq->tokens.end());
    completed.push_back(std::move(c));

    // Freed here rather than at the next step boundary: the blocks are dead the
    // moment the sequence is, and holding them would refuse admission to a
    // request that could have used them.
    if (seq->table != nullptr) {
      (void)pool->FreeBlocks(seq->table->blocks());
      seq->table->Clear();
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

      // Reserved for the prompt only. Growth for generated tokens happens per
      // step, which is what makes the pool shared rather than partitioned.
      const Status reserved = Reserve(&seq, needed);

      if (!reserved.ok()) {
        // Not enough blocks right now. Put back what was taken and leave the
        // request queued -- a running sequence finishing will free room, and
        // FCFS means this one goes first when it does.
        (void)pool->FreeBlocks(seq.table->blocks());
        seq.table->Clear();
        waiting.front() = std::move(seq);
        return;
      }

      waiting.pop_front();
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

  if (impl_->running.empty()) return OkStatus();

  const int64_t num_seqs = static_cast<int64_t>(impl_->running.size());

  out_batch->num_seqs = num_seqs;
  out_batch->max_blocks_per_seq = impl_->max_blocks_per_seq;
  out_batch->block_table.assign(
      static_cast<size_t>(num_seqs * impl_->max_blocks_per_seq), 0);
  out_batch->seq_lens.assign(static_cast<size_t>(num_seqs), 0);

  // ---- Plan ----
  //
  // Decided for the whole batch before anything is emitted, because the token
  // budget is handed out in a different order than the batch is written in.
  // Emission has to run in `running` order -- the attention plan requires rows
  // grouped by sequence and in ascending sequence index -- while the budget
  // goes to decodes first regardless of where they sit in that order.
  impl_->take.assign(static_cast<size_t>(num_seqs), 0);
  int64_t budget = impl_->config.max_batch_tokens;

  // Decodes first (§8.1). One token each, and they are served ahead of any
  // prompt so that a decoding sequence's inter-token time does not include
  // some other request's prefill. That bound is the entire reason chunking
  // exists, so it is the priority rule and not a tuning choice.
  //
  // A prompt's final chunk has one token pending too, and lands here. That is
  // right: it is one token of work either way, and it is the step that request
  // has been waiting through its whole prefill for.
  for (int64_t s = 0; s < num_seqs && budget > 0; ++s) {
    if (impl_->running[static_cast<size_t>(s)].pending() == 1) {
      impl_->take[static_cast<size_t>(s)] = 1;
      --budget;
    }
  }

  // Then prompts fill what is left, FCFS so the oldest finishes first (§8.3).
  // A prompt that does not fit is *chunked* rather than deferred: it takes the
  // rest of the budget now and resumes next step. The alternative -- waiting
  // for a step it fits in whole -- is what made a long prompt able to stall
  // behind a busy engine indefinitely, and what made a prompt longer than the
  // budget impossible to serve at all.
  for (int64_t s = 0; s < num_seqs && budget > 0; ++s) {
    const int64_t pending = impl_->running[static_cast<size_t>(s)].pending();
    if (pending <= 1) continue;

    const int64_t chunk = std::min(pending, budget);
    impl_->take[static_cast<size_t>(s)] = chunk;
    budget -= chunk;
  }

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
    if (contributes) {
      INFERX_RETURN_IF_ERROR(impl_->Reserve(&seq, seq.cached + take));
    }

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
      out_batch->seeds.push_back(seq.params.seed ^
                                 (0x9e3779b97f4a7c15ULL *
                                  static_cast<uint64_t>(seq.tokens.size())));
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
    return InvalidArgumentError("got ", sampled.size(), " sampled tokens but "
                                "the batch asked for ", expected);
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
  return impl_->pool->used_blocks();
}

}  // namespace inferx::scheduler
