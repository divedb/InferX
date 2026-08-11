#include "inferx/model/weight_loader.h"

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <utility>

#include "inferx/core/arena.h"
#include "inferx/core/device_runtime.h"
#include "inferx/core/dtype.h"
#include "inferx/core/tensor.h"

namespace inferx::model {
namespace {

using Clock = std::chrono::steady_clock;

double SecondsSince(Clock::time_point start) {
  return std::chrono::duration_cast<std::chrono::duration<double>>(
             Clock::now() - start)
      .count();
}

// `len` contiguous source bytes destined for the running output offset. Every
// verb lowers to a sequence of these; the streaming loop below is the only
// code that knows about slots, threads, or the copy engine.
struct Extent {
  const std::byte* src = nullptr;
  size_t len = 0;
};

// One packed piece of a staging slot: `len` bytes from `src` to `dst_off`
// within the slot. Kept at or below the stripe size so a parallel fill
// balances even when the slot is a single large extent.
struct Segment {
  const std::byte* src = nullptr;
  size_t dst_off = 0;
  size_t len = 0;
};

// Below this, splitting a copy across threads costs more than it saves.
constexpr size_t kStripeBytes = size_t{1} << 20;

// Persistent workers for the host side of staging. The page faults a cold
// load spends most of its time in are taken inside these memcpys, so N
// threads fault N streams of the file in parallel — the actual win; the copy
// engine needs no threads at all. `Run` also works from the calling thread,
// so a pool built with 0 workers still runs everything, just serially.
class CopyPool {
 public:
  explicit CopyPool(int workers) {
    workers_.reserve(static_cast<size_t>(workers));
    for (int i = 0; i < workers; ++i) {
      workers_.emplace_back([this] { WorkerLoop(); });
    }
  }

  ~CopyPool() {
    {
      std::lock_guard<std::mutex> lock(mu_);
      stop_ = true;
    }
    wake_.notify_all();
    for (std::thread& t : workers_) t.join();
  }

  // Runs fn(0..jobs-1), returning when all are done.
  void Run(int jobs, const std::function<void(int)>& fn) {
    if (jobs <= 0) return;
    {
      std::lock_guard<std::mutex> lock(mu_);
      fn_ = &fn;
      next_ = 0;
      jobs_ = jobs;
      pending_ = jobs;
      ++generation_;
    }
    wake_.notify_all();

    Drain();

    std::unique_lock<std::mutex> lock(mu_);
    done_.wait(lock, [this] { return pending_ == 0; });
    fn_ = nullptr;
  }

 private:
  // Claims and runs jobs until none remain. Callable from the submitting
  // thread and from workers; `fn_` is only dereferenced by a thread that has
  // claimed a job under the lock, and `pending_` reaching zero proves every
  // such dereference has returned.
  void Drain() {
    std::unique_lock<std::mutex> lock(mu_);
    while (next_ < jobs_) {
      const int job = next_++;
      const std::function<void(int)>* fn = fn_;
      lock.unlock();
      (*fn)(job);
      lock.lock();
      if (--pending_ == 0) done_.notify_all();
    }
  }

  void WorkerLoop() {
    uint64_t seen = 0;
    while (true) {
      {
        std::unique_lock<std::mutex> lock(mu_);
        wake_.wait(lock, [&] {
          return stop_ || (generation_ != seen && next_ < jobs_);
        });
        if (stop_) return;
        seen = generation_;
      }
      Drain();
    }
  }

  std::mutex mu_;
  std::condition_variable wake_;
  std::condition_variable done_;
  std::vector<std::thread> workers_;
  const std::function<void(int)>* fn_ = nullptr;
  int next_ = 0;
  int jobs_ = 0;
  int pending_ = 0;
  uint64_t generation_ = 0;
  bool stop_ = false;
};

int ResolveThreads(int requested) {
  if (requested > 0) return requested;
  const unsigned hw = std::thread::hardware_concurrency();
  return static_cast<int>(std::clamp(hw / 2, 2u, 8u));
}

}  // namespace

struct WeightLoader::Impl {
  const Checkpoint* ckpt = nullptr;
  Options opts;
  int threads = 1;
  std::unique_ptr<CopyPool> pool;

  // The pinned ring. Empty in CPU mode, where verbs gather straight into the
  // destination and none of the CUDA runtime is touched.
  struct Slot {
    std::byte* pinned = nullptr;
    bool in_flight = false;
    DeviceEvent event;
  };
  std::vector<Slot> slots;
  size_t next_slot = 0;
  DeviceRuntime* runtime = nullptr;
  Stream copy_stream;

  std::vector<DeviceBuffer> chunks;
  BumpArena arena;

  WeightLoaderStats stats;

  // Scratch reused across calls to avoid per-slot allocation.
  std::vector<Segment> segments;
  std::vector<size_t> job_bounds;
  std::vector<Extent> extents;

  StatusOr<std::byte*> AllocateOut(size_t bytes);
  Status TransferExtents(absl::Span<const Extent> extents_in, std::byte* dst,
                         size_t total);
  StatusOr<TensorView> FinishLoad(absl::Span<const Extent> extents_in,
                                  size_t total, DataType dtype,
                                  const Shape& shape);

  ~Impl() {
    if (runtime != nullptr && copy_stream.handle != nullptr) {
      // In-flight copies read the pinned slots; drain before freeing them.
      (void)runtime->SynchronizeStream(copy_stream);
      (void)runtime->DestroyStream(copy_stream);
    }
    for (Slot& slot : slots) {
      if (slot.event.handle != nullptr) (void)runtime->DestroyEvent(slot.event);
      if (slot.pinned != nullptr) (void)runtime->FreePinnedHost(slot.pinned);
    }
  }
};

// Carves `bytes` of output memory from the current chunk, opening a new one
// when it does not fit. Oversized requests get a dedicated buffer: the
// alternative is a chunk-sized hole.
StatusOr<std::byte*> WeightLoader::Impl::AllocateOut(size_t bytes) {
  if (bytes > opts.device_chunk_bytes) {
    INFERX_ASSIGN_OR_RETURN(DeviceBuffer buf,
                            DeviceBuffer::Allocate(bytes, opts.device));
    std::byte* out = buf.data();
    chunks.push_back(std::move(buf));
    return out;
  }

  if (arena.Remaining() < bytes + kTensorAlignment) {
    INFERX_ASSIGN_OR_RETURN(
        DeviceBuffer buf,
        DeviceBuffer::Allocate(opts.device_chunk_bytes, opts.device));
    arena = BumpArena(buf.data(), buf.size(), opts.device);
    chunks.push_back(std::move(buf));
  }

  INFERX_ASSIGN_OR_RETURN(void* out, arena.Allocate(bytes));
  return static_cast<std::byte*>(out);
}

// Streams `extents_in` (`total` bytes) to `dst` through the pinned ring: pack
// a slot with the worker pool, hand it to the copy engine, move on. In CPU
// mode the "slot" is the destination itself and the copy is one pass.
Status WeightLoader::Impl::TransferExtents(absl::Span<const Extent> extents_in,
                                           std::byte* dst, size_t total) {
  const bool accelerator = opts.device.IsAccelerator();
  const size_t slot_bytes = accelerator ? opts.staging_slot_bytes : total;

  size_t ei = 0;    // current extent
  size_t eoff = 0;  // bytes of it already consumed
  size_t out_off = 0;

  while (out_off < total) {
    const size_t fill = std::min(slot_bytes, total - out_off);

    std::byte* base = dst + out_off;
    Slot* slot = nullptr;
    if (accelerator) {
      slot = &slots[next_slot];
      next_slot = (next_slot + 1) % slots.size();
      if (slot->in_flight) {
        const auto wait_start = Clock::now();
        INFERX_RETURN_IF_ERROR(runtime->SynchronizeEvent(slot->event));
        stats.h2d_wait_seconds += SecondsSince(wait_start);
        slot->in_flight = false;
      }
      base = slot->pinned;
    }

    // Split the extents covering this slot at the slot boundary and the
    // stripe size, then partition the pieces among the threads by bytes.
    segments.clear();
    size_t off = 0;
    while (off < fill) {
      const Extent& e = extents_in[ei];
      const size_t take = std::min({e.len - eoff, fill - off, kStripeBytes});
      segments.push_back(Segment{e.src + eoff, off, take});
      off += take;
      eoff += take;
      if (eoff == e.len) {
        ++ei;
        eoff = 0;
      }
    }

    const int jobs = static_cast<int>(
        std::min<size_t>(static_cast<size_t>(threads), segments.size()));
    job_bounds.assign(static_cast<size_t>(jobs) + 1, segments.size());
    job_bounds[0] = 0;
    const size_t per_job =
        (fill + static_cast<size_t>(jobs) - 1) / static_cast<size_t>(jobs);
    size_t acc = 0;
    for (size_t s = 0, j = 1; s < segments.size() && j + 1 < job_bounds.size();
         ++s) {
      acc += segments[s].len;
      if (acc >= per_job * j) job_bounds[j++] = s + 1;
    }

    const auto stage_start = Clock::now();
    pool->Run(jobs, [&](int j) {
      for (size_t s = job_bounds[static_cast<size_t>(j)];
           s < job_bounds[static_cast<size_t>(j) + 1]; ++s) {
        std::memcpy(base + segments[s].dst_off, segments[s].src,
                    segments[s].len);
      }
    });
    stats.stage_seconds += SecondsSince(stage_start);

    if (accelerator) {
      INFERX_RETURN_IF_ERROR(runtime->CopyAsync(
          dst + out_off, base, fill, CopyKind::kHostToDevice, copy_stream));
      INFERX_RETURN_IF_ERROR(runtime->RecordEvent(slot->event, copy_stream));
      slot->in_flight = true;
    }

    out_off += fill;
  }

  return OkStatus();
}

StatusOr<TensorView> WeightLoader::Impl::FinishLoad(
    absl::Span<const Extent> extents_in, size_t total, DataType dtype,
    const Shape& shape) {
  INFERX_ASSIGN_OR_RETURN(std::byte * dst, AllocateOut(total));
  INFERX_RETURN_IF_ERROR(TransferExtents(extents_in, dst, total));
  ++stats.tensors;
  stats.bytes += total;
  return TensorView::Create(dst, dtype, shape, opts.device);
}

StatusOr<WeightLoader> WeightLoader::Create(const Checkpoint* checkpoint,
                                            const Options& options) {
  if (checkpoint == nullptr) {
    return InvalidArgumentError("WeightLoader: null checkpoint");
  }

  auto impl = std::make_unique<Impl>();
  impl->ckpt = checkpoint;
  impl->opts = options;
  impl->opts.staging_slot_bytes =
      std::max(options.staging_slot_bytes, kStripeBytes);
  impl->opts.staging_slots = std::max(options.staging_slots, 2);
  impl->threads = ResolveThreads(options.threads);
  impl->pool = std::make_unique<CopyPool>(impl->threads - 1);

  INFERX_ASSIGN_OR_RETURN(impl->runtime, RuntimeFor(impl->opts.device));
  if (impl->opts.device.IsAccelerator()) {
    if (!impl->runtime->capabilities().pinned_host_memory) {
      return UnimplementedError("WeightLoader: ", impl->opts.device.ToString(),
                                " runtime has no pinned host memory");
    }
    INFERX_ASSIGN_OR_RETURN(impl->copy_stream,
                            impl->runtime->CreateStream(impl->opts.device));
    impl->slots.resize(static_cast<size_t>(impl->opts.staging_slots));
    for (Impl::Slot& slot : impl->slots) {
      INFERX_ASSIGN_OR_RETURN(void* pinned, impl->runtime->AllocatePinnedHost(
                                                impl->opts.staging_slot_bytes));
      slot.pinned = static_cast<std::byte*>(pinned);
      INFERX_ASSIGN_OR_RETURN(slot.event,
                              impl->runtime->CreateEvent(/*timing=*/false));
    }
  }

  return WeightLoader(std::move(impl));
}

StatusOr<TensorView> WeightLoader::Load(std::string_view name,
                                        const Shape& expected) {
  INFERX_ASSIGN_OR_RETURN(const Tensor t,
                          impl_->ckpt->GetChecked(name, expected));
  const size_t total = static_cast<size_t>(t.nbytes());
  const Extent extent{static_cast<const std::byte*>(t.data()), total};
  return impl_->FinishLoad(absl::MakeConstSpan(&extent, 1), total, t.dtype(),
                           expected);
}

StatusOr<TensorView> WeightLoader::Load(std::string_view name,
                                        const Shape& expected,
                                        const parallel::ShardSpec& shard,
                                        int tp_rank, int tp_size) {
  if (tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size) {
    return InvalidArgumentError("Load: invalid TP topology rank=", tp_rank,
                                " size=", tp_size);
  }
  if (shard.partition == parallel::Partition::kReplicated || tp_size == 1) {
    return Load(name, expected);
  }
  if (shard.unit <= 0) {
    return InvalidArgumentError("Load: shard unit must be positive for ", name);
  }

  INFERX_ASSIGN_OR_RETURN(const Tensor tensor,
                          impl_->ckpt->GetChecked(name, expected));
  if (DataTypeBitWidth(tensor.dtype()) % 8 != 0) {
    return UnimplementedError("Load: packed dtype ",
                              DataTypeName(tensor.dtype()),
                              " cannot be sharded before unpacking");
  }
  INFERX_ASSIGN_OR_RETURN(const int axis,
                          parallel::ResolveShardAxis(shard, tensor.rank()));
  const int64_t global_axis = expected.Dim(axis);
  if (shard.segments <= 0 || global_axis % shard.segments != 0) {
    return InvalidArgumentError("Load: dimension ", global_axis, " of ", name,
                                " cannot split into ", shard.segments,
                                " shard segments");
  }
  const int64_t segment_axis = global_axis / shard.segments;
  if (segment_axis % tp_size != 0) {
    return InvalidArgumentError("Load: dimension ", global_axis, " on axis ",
                                axis, " of ", name,
                                " has segments that are not divisible by TP "
                                "size ",
                                tp_size);
  }
  const int64_t local_segment = segment_axis / tp_size;
  const int64_t local_axis = local_segment * shard.segments;
  if (local_segment % shard.unit != 0) {
    return InvalidArgumentError("Load: local segment ", local_segment, " of ",
                                name, " is not aligned to shard unit ",
                                shard.unit);
  }

  Shape local_shape = expected;
  local_shape.SetDim(axis, local_axis);
  const int64_t inner = expected.InnerNumel(axis) / global_axis;
  const int64_t outer = expected.Numel() / (global_axis * inner);
  const size_t element_bytes = DataTypeBitWidth(tensor.dtype()) / 8;
  const size_t extent_bytes =
      static_cast<size_t>(local_segment * inner) * element_bytes;
  const size_t segment_stride =
      static_cast<size_t>(segment_axis * inner) * element_bytes;
  const size_t source_stride =
      static_cast<size_t>(global_axis * inner) * element_bytes;
  const size_t rank_offset = static_cast<size_t>(tp_rank) * extent_bytes;
  const auto* source = static_cast<const std::byte*>(tensor.data());

  std::vector<Extent>& extents = impl_->extents;
  extents.clear();
  extents.reserve(static_cast<size_t>(outer * shard.segments));
  for (int64_t i = 0; i < outer; ++i) {
    for (int segment = 0; segment < shard.segments; ++segment) {
      extents.push_back(Extent{
          source + static_cast<size_t>(i) * source_stride +
              static_cast<size_t>(segment) * segment_stride + rank_offset,
          extent_bytes});
    }
  }
  const size_t total = static_cast<size_t>(
      DataTypeByteSize(tensor.dtype(), local_shape.Numel()));
  return impl_->FinishLoad(extents, total, tensor.dtype(), local_shape);
}

StatusOr<TensorView> WeightLoader::Load(std::string_view name) {
  INFERX_ASSIGN_OR_RETURN(const Tensor t, impl_->ckpt->Get(name));
  const size_t total = static_cast<size_t>(t.nbytes());
  const Extent extent{static_cast<const std::byte*>(t.data()), total};
  return impl_->FinishLoad(absl::MakeConstSpan(&extent, 1), total, t.dtype(),
                           t.shape());
}

StatusOr<TensorView> WeightLoader::LoadStacked(
    absl::Span<const std::string> names, const Shape& part, const Shape& out) {
  if (names.empty()) {
    return InvalidArgumentError("LoadStacked: no source tensors");
  }

  std::vector<Tensor> parts;  // keeps the mmap borrows alive through staging
  parts.reserve(names.size());
  for (const std::string& name : names) {
    INFERX_ASSIGN_OR_RETURN(Tensor t, impl_->ckpt->GetChecked(name, part));
    parts.push_back(std::move(t));
  }

  return UploadStacked(parts, out);
}

StatusOr<TensorView> WeightLoader::LoadStacked(
    absl::Span<const std::string> names, absl::Span<const Shape> global_parts,
    const Shape& global_out, const parallel::ShardSpec& shard, int tp_rank,
    int tp_size) {
  if (names.empty() || names.size() != global_parts.size()) {
    return InvalidArgumentError(
        "LoadStacked: names and shapes must have the "
        "same non-zero size");
  }
  if (tp_size <= 0 || tp_rank < 0 || tp_rank >= tp_size) {
    return InvalidArgumentError(
        "LoadStacked: invalid TP topology rank=", tp_rank, " size=", tp_size);
  }
  if (shard.partition == parallel::Partition::kReplicated || tp_size == 1) {
    std::vector<Tensor> parts;
    parts.reserve(names.size());
    for (size_t i = 0; i < names.size(); ++i) {
      INFERX_ASSIGN_OR_RETURN(
          Tensor tensor, impl_->ckpt->GetChecked(names[i], global_parts[i]));
      parts.push_back(std::move(tensor));
    }
    return UploadStacked(parts, global_out);
  }
  if (shard.unit <= 0) {
    return InvalidArgumentError("LoadStacked: shard unit must be positive");
  }

  std::vector<Tensor> parts;
  parts.reserve(names.size());
  std::vector<Extent>& extents = impl_->extents;
  extents.clear();
  Shape local_out = global_out;
  INFERX_ASSIGN_OR_RETURN(const int out_axis,
                          parallel::ResolveShardAxis(shard, global_out.Rank()));
  if (global_out.Dim(out_axis) % tp_size != 0) {
    return InvalidArgumentError("LoadStacked: output dimension ",
                                global_out.Dim(out_axis), " on axis ", out_axis,
                                " is not divisible by TP size ", tp_size);
  }
  local_out.SetDim(out_axis, global_out.Dim(out_axis) / tp_size);

  DataType dtype = DataType::kUndefined;
  size_t total = 0;
  for (size_t part_index = 0; part_index < names.size(); ++part_index) {
    INFERX_ASSIGN_OR_RETURN(
        Tensor tensor,
        impl_->ckpt->GetChecked(names[part_index], global_parts[part_index]));
    if (part_index == 0) {
      dtype = tensor.dtype();
    } else if (tensor.dtype() != dtype) {
      return InvalidArgumentError("LoadStacked: mixed dtypes");
    }
    if (DataTypeBitWidth(dtype) % 8 != 0) {
      return UnimplementedError("LoadStacked: packed dtype ",
                                DataTypeName(dtype),
                                " cannot be sharded before unpacking");
    }
    INFERX_ASSIGN_OR_RETURN(const int axis,
                            parallel::ResolveShardAxis(shard, tensor.rank()));
    const int64_t global_axis = tensor.dim(axis);
    if (shard.segments <= 0 || global_axis % shard.segments != 0) {
      return InvalidArgumentError(
          "LoadStacked: dimension ", global_axis, " of ", names[part_index],
          " cannot split into ", shard.segments, " shard segments");
    }
    const int64_t segment_axis = global_axis / shard.segments;
    if (segment_axis % tp_size != 0) {
      return InvalidArgumentError("LoadStacked: dimension ", global_axis,
                                  " on axis ", axis, " of ", names[part_index],
                                  " has segments that are not divisible by TP "
                                  "size ",
                                  tp_size);
    }
    const int64_t local_segment = segment_axis / tp_size;
    if (local_segment % shard.unit != 0) {
      return InvalidArgumentError("LoadStacked: local segment ", local_segment,
                                  " of ", names[part_index],
                                  " is not aligned to shard unit ", shard.unit);
    }
    const int64_t inner = tensor.shape().InnerNumel(axis) / global_axis;
    const int64_t outer = tensor.numel() / (global_axis * inner);
    const size_t element_bytes = DataTypeBitWidth(dtype) / 8;
    const size_t extent_bytes =
        static_cast<size_t>(local_segment * inner) * element_bytes;
    const size_t segment_stride =
        static_cast<size_t>(segment_axis * inner) * element_bytes;
    const size_t source_stride =
        static_cast<size_t>(global_axis * inner) * element_bytes;
    const size_t rank_offset = static_cast<size_t>(tp_rank) * extent_bytes;
    const auto* source = static_cast<const std::byte*>(tensor.data());
    for (int64_t i = 0; i < outer; ++i) {
      for (int segment = 0; segment < shard.segments; ++segment) {
        extents.push_back(Extent{
            source + static_cast<size_t>(i) * source_stride +
                static_cast<size_t>(segment) * segment_stride + rank_offset,
            extent_bytes});
        total += extent_bytes;
      }
    }
    parts.push_back(std::move(tensor));
  }
  if (total !=
      static_cast<size_t>(DataTypeByteSize(dtype, local_out.Numel()))) {
    return InvalidArgumentError("LoadStacked: local shards do not fill ",
                                local_out.ToString());
  }
  return impl_->FinishLoad(extents, total, dtype, local_out);
}

StatusOr<TensorView> WeightLoader::Upload(const Tensor& host) {
  if (!host.defined() || !host.is_cpu()) {
    return InvalidArgumentError("Upload: source must be a defined host tensor");
  }
  const size_t total = static_cast<size_t>(host.nbytes());
  const Extent extent{static_cast<const std::byte*>(host.data()), total};
  return impl_->FinishLoad(absl::MakeConstSpan(&extent, 1), total, host.dtype(),
                           host.shape());
}

StatusOr<TensorView> WeightLoader::UploadStacked(absl::Span<const Tensor> parts,
                                                 const Shape& out) {
  if (parts.empty()) {
    return InvalidArgumentError("UploadStacked: no source tensors");
  }

  std::vector<Extent>& extents = impl_->extents;
  extents.clear();

  size_t total = 0;
  for (const Tensor& t : parts) {
    if (!t.defined() || !t.is_cpu()) {
      return InvalidArgumentError(
          "UploadStacked: sources must be defined host tensors");
    }
    if (t.dtype() != parts.front().dtype()) {
      return InvalidArgumentError("UploadStacked: mixed dtypes ",
                                  DataTypeName(t.dtype()), " and ",
                                  DataTypeName(parts.front().dtype()));
    }
    extents.push_back(Extent{static_cast<const std::byte*>(t.data()),
                             static_cast<size_t>(t.nbytes())});
    total += static_cast<size_t>(t.nbytes());
  }

  const DataType dtype = parts.front().dtype();
  if (total != static_cast<size_t>(DataTypeByteSize(dtype, out.Numel()))) {
    return InvalidArgumentError("UploadStacked: ", parts.size(),
                                " parts totalling ", total,
                                " bytes do not fill ", out.ToString());
  }

  return impl_->FinishLoad(extents, total, dtype, out);
}

StatusOr<TensorView> WeightLoader::LoadRowPermuted(
    std::string_view name, const Shape& expected,
    absl::Span<const int64_t> row_map) {
  if (expected.Rank() != 2) {
    return InvalidArgumentError("LoadRowPermuted: ", name, " must be 2-D, got ",
                                expected.ToString());
  }
  INFERX_ASSIGN_OR_RETURN(const Tensor t,
                          impl_->ckpt->GetChecked(name, expected));

  const int64_t rows = t.dim(0);
  if (static_cast<int64_t>(row_map.size()) != rows) {
    return InvalidArgumentError("LoadRowPermuted: row_map covers ",
                                row_map.size(), " of ", rows, " rows of ",
                                name);
  }

  const size_t row_bytes = DataTypeByteSize(t.dtype(), t.dim(1));
  const auto* src = static_cast<const std::byte*>(t.data());
  std::vector<Extent>& extents = impl_->extents;
  extents.clear();
  extents.reserve(row_map.size());
  for (const int64_t r : row_map) {
    if (r < 0 || r >= rows) {
      return InvalidArgumentError("LoadRowPermuted: row ", r,
                                  " out of range for ", name);
    }
    extents.push_back(
        Extent{src + static_cast<size_t>(r) * row_bytes, row_bytes});
  }

  return impl_->FinishLoad(extents, static_cast<size_t>(t.nbytes()), t.dtype(),
                           expected);
}

Status WeightLoader::Finish() {
  if (impl_->copy_stream.handle != nullptr) {
    const auto wait_start = Clock::now();
    INFERX_RETURN_IF_ERROR(
        impl_->runtime->SynchronizeStream(impl_->copy_stream));
    impl_->stats.h2d_wait_seconds += SecondsSince(wait_start);
    for (Impl::Slot& slot : impl_->slots) slot.in_flight = false;
  }
  return OkStatus();
}

StatusOr<std::vector<DeviceBuffer>> WeightLoader::Release() {
  INFERX_RETURN_IF_ERROR(Finish());
  impl_->arena = BumpArena();
  return std::move(impl_->chunks);
}

const WeightLoaderStats& WeightLoader::stats() const { return impl_->stats; }

int WeightLoader::threads() const { return impl_->threads; }

size_t WeightLoader::buffer_count() const { return impl_->chunks.size(); }

StatusOr<Shape> ConcatenatedShape(absl::Span<const Tensor> parts) {
  if (parts.empty()) {
    return InvalidArgumentError("ConcatenatedShape: no parts");
  }

  int64_t rows = 0;
  const int64_t cols = parts.front().rank() == 2 ? parts.front().dim(1) : 0;

  for (const Tensor& t : parts) {
    rows += t.dim(0);
    if (t.rank() == 2 && t.dim(1) != cols) {
      return InvalidArgumentError("cannot concatenate: widths ", cols, " and ",
                                  t.dim(1), " differ");
    }
  }
  return parts.front().rank() == 2 ? Shape({rows, cols}) : Shape({rows});
}

WeightLoader::WeightLoader(std::unique_ptr<Impl> impl)
    : impl_(std::move(impl)) {}
WeightLoader::~WeightLoader() = default;
WeightLoader::WeightLoader(WeightLoader&&) noexcept = default;
WeightLoader& WeightLoader::operator=(WeightLoader&&) noexcept = default;

}  // namespace inferx::model
