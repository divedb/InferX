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

  /// Tokens waiting to be run: the prompt on the first step, one token after.
  int64_t pending() const {
    return static_cast<int64_t>(tokens.size()) - cached;
  }
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

  /// Which running index each `logits_indices` entry came from, so `CommitStep`
  /// can put a sampled token back where it belongs. Rebuilt every step.
  std::vector<int64_t> batch_seq_index;

  int64_t max_blocks_per_seq = 0;

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
  impl_->batch_seq_index.clear();

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

  out_batch->num_seqs = static_cast<int64_t>(impl_->running.size());
  out_batch->max_blocks_per_seq = impl_->max_blocks_per_seq;
  out_batch->block_table.assign(
      static_cast<size_t>(out_batch->num_seqs * impl_->max_blocks_per_seq), 0);

  int64_t budget = impl_->config.max_batch_tokens;

  for (size_t s = 0; s < impl_->running.size(); ++s) {
    Sequence& seq = impl_->running[s];

    // A sequence's row in the block table, regardless of whether it contributes
    // tokens this step -- the row index is its identity for the whole batch.
    for (int64_t b = 0; b < seq.table->size(); ++b) {
      out_batch->block_table[static_cast<size_t>(
          static_cast<int64_t>(s) * impl_->max_blocks_per_seq + b)] =
          seq.table->blocks()[static_cast<size_t>(b)];
    }

    const int64_t pending = seq.pending();
    if (pending <= 0) continue;

    // No chunking yet (M5): a sequence either fits in what is left of the
    // budget or waits for the next step. Decodes are one token and always fit,
    // so this only ever defers a prefill.
    if (pending > budget) continue;

    INFERX_RETURN_IF_ERROR(
        impl_->Reserve(&seq, static_cast<int64_t>(seq.tokens.size())));

    const int64_t start = seq.cached;

    for (int64_t i = start; i < static_cast<int64_t>(seq.tokens.size()); ++i) {
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

    budget -= pending;

    // Only the last token of each sequence produces logits: the earlier ones
    // are prompt, and their predictions are discarded.
    out_batch->logits_indices.push_back(
        static_cast<int32_t>(out_batch->token_ids.size() - 1));
    impl_->batch_seq_index.push_back(static_cast<int64_t>(s));
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

  if (sampled.size() != impl_->batch_seq_index.size()) {
    return InvalidArgumentError("got ", sampled.size(), " sampled tokens but "
                                "the batch asked for ",
                                impl_->batch_seq_index.size());
  }

  for (size_t i = 0; i < sampled.size(); ++i) {
    Sequence& seq =
        impl_->running[static_cast<size_t>(impl_->batch_seq_index[i])];

    // Everything that went into the batch is now cached, including the tokens
    // whose logits were discarded.
    seq.cached = static_cast<int64_t>(seq.tokens.size());

    seq.tokens.push_back(sampled[i]);

    const bool is_stop =
        std::find(seq.params.stop_tokens.begin(), seq.params.stop_tokens.end(),
                  sampled[i]) != seq.params.stop_tokens.end();

    if (is_stop) {
      seq.finish = FinishReason::kStopToken;
    } else if (seq.generated() >= seq.params.max_tokens) {
      seq.finish = FinishReason::kMaxTokens;
    }

    // Emitted after the stop checks, so `finish` is already decided and a
    // streaming caller can close the stream on this same delta.
    if (out_deltas != nullptr) {
      out_deltas->push_back({seq.id, sampled[i], seq.finish});
    }
  }

  impl_->batch_seq_index.clear();

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
