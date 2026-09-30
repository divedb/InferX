#include "inferx/dist/nccl_comm.h"

#include <dlfcn.h>

#include <condition_variable>
#include <cstring>
#include <mutex>
#include <optional>
#include <utility>

#include "inferx/core/logging.h"
#include "inferx/ops/op_context.h"

namespace inferx {
namespace dist {
namespace {

// ncclResult_t values this code distinguishes.
constexpr int kNcclSuccess = 0;

// ncclDataType_t: the subset used here. NCCL's enum layout is stable ABI.
constexpr int kNcclBfloat16 = 9;

// ncclRedOp_t.
constexpr int kNcclSum = 0;

// ncclCommInitRank takes this struct by value, not a pointer to its bytes.
struct NcclIdAbi {
  char internal[128];
};
static_assert(sizeof(NcclIdAbi) == sizeof(NcclUniqueId));

/// \brief The libnccl function table, loaded once per process.
struct NcclApi {
  void* library = nullptr;
  int (*comm_init_rank)(void**, int, NcclIdAbi, int) = nullptr;
  int (*comm_destroy)(void*) = nullptr;
  int (*all_reduce)(const void*, void*, size_t, int, int, void*, void*) = nullptr;
  int (*all_gather)(const void*, void*, size_t, int, void*, void*) = nullptr;
  int (*get_unique_id)(NcclIdAbi*) = nullptr;
  int (*comm_init_all)(void**, int, const int*) = nullptr;
  int (*comm_abort)(void*) = nullptr;
  int (*get_async_error)(void*, int*) = nullptr;
  int (*group_start)() = nullptr;
  int (*group_end)() = nullptr;

  static const NcclApi* Load() {
    static const NcclApi* api = []() -> NcclApi* {
      auto* loaded = new NcclApi;
      // Mirror NCCL's ABI without requiring its headers at build time.
      loaded->library = dlopen("libnccl.so.2", RTLD_NOW | RTLD_GLOBAL);
      if (loaded->library == nullptr) {
        loaded->library = dlopen("libnccl.so", RTLD_NOW | RTLD_GLOBAL);
      }
      if (loaded->library == nullptr) return loaded;
      loaded->comm_init_rank = reinterpret_cast<int (*)(void**, int, NcclIdAbi, int)>(
          dlsym(loaded->library, "ncclCommInitRank"));
      loaded->comm_destroy =
          reinterpret_cast<int (*)(void*)>(dlsym(loaded->library, "ncclCommDestroy"));
      loaded->all_reduce =
          reinterpret_cast<int (*)(const void*, void*, size_t, int, int, void*, void*)>(
              dlsym(loaded->library, "ncclAllReduce"));
      loaded->all_gather =
          reinterpret_cast<int (*)(const void*, void*, size_t, int, void*, void*)>(
              dlsym(loaded->library, "ncclAllGather"));
      loaded->get_unique_id =
          reinterpret_cast<int (*)(NcclIdAbi*)>(dlsym(loaded->library, "ncclGetUniqueId"));
      loaded->comm_init_all = reinterpret_cast<int (*)(void**, int, const int*)>(
          dlsym(loaded->library, "ncclCommInitAll"));
      loaded->comm_abort =
          reinterpret_cast<int (*)(void*)>(dlsym(loaded->library, "ncclCommAbort"));
      loaded->get_async_error = reinterpret_cast<int (*)(void*, int*)>(
          dlsym(loaded->library, "ncclCommGetAsyncError"));
      loaded->group_start =
          reinterpret_cast<int (*)()>(dlsym(loaded->library, "ncclGroupStart"));
      loaded->group_end = reinterpret_cast<int (*)()>(dlsym(loaded->library, "ncclGroupEnd"));
      return loaded;
    }();
    return api;
  }

  bool Ready() const {
    return library != nullptr && comm_init_rank != nullptr && comm_destroy != nullptr &&
           all_reduce != nullptr && all_gather != nullptr && get_unique_id != nullptr;
  }
};

Status Unavailable() {
  return NotFoundError(
      "libnccl is not available; tensor-parallel worlds beyond one "
      "process need an NCCL installation");
}

Status NcclCall(int result, const char* what) {
  if (result == kNcclSuccess) return OkStatus();
  return InternalError(what, " failed with nccl result ", result);
}

}  // namespace

StatusOr<NcclUniqueId> NcclComm::NewUniqueId() {
  const NcclApi* api = NcclApi::Load();
  if (!api->Ready()) return Unavailable();
  NcclUniqueId id{};
  NcclIdAbi native_id{};
  INFERX_RETURN_IF_ERROR(NcclCall(api->get_unique_id(&native_id), "ncclGetUniqueId"));
  std::memcpy(id.data(), native_id.internal, id.size());
  return id;
}

StatusOr<std::unique_ptr<NcclComm>> NcclComm::Join(int rank, int size, const NcclUniqueId& id,
                                                   DeviceId device) {
  if (rank < 0 || size <= 0 || rank >= size) {
    return InvalidArgumentError("invalid NCCL topology: rank ", rank, " of ", size);
  }
  if (!device.IsCuda()) return InvalidArgumentError("NCCL requires a CUDA device");
  const NcclApi* api = NcclApi::Load();
  if (!api->Ready()) return Unavailable();
  INFERX_ASSIGN_OR_RETURN(auto runtime, RuntimeFor(device));
  INFERX_RETURN_IF_ERROR(runtime->Activate());
  NcclIdAbi native_id{};
  std::memcpy(native_id.internal, id.data(), id.size());
  void* comm = nullptr;
  INFERX_RETURN_IF_ERROR(
      NcclCall(api->comm_init_rank(&comm, size, native_id, rank), "ncclCommInitRank"));
  return std::unique_ptr<NcclComm>(new NcclComm(rank, size, comm, device));
}

NcclComm::NcclComm(int rank, int size, void* comm, DeviceId device)
    : rank_(rank), size_(size), comm_(comm), device_(device) {}

NcclComm::~NcclComm() {
  const NcclApi* api = NcclApi::Load();
  if (comm_ != nullptr && api->comm_destroy != nullptr) {
    if (auto runtime = RuntimeFor(device_); runtime.ok()) (void)(*runtime)->Activate();
    if (const int rc = api->comm_destroy(comm_); rc != kNcclSuccess) {
      INFERX_LOG(ERROR) << "ncclCommDestroy failed with result " << rc;
    }
  }
}

Status NcclComm::AllReduceSum(const ops::OpContext& ctx, Tensor& partial) {
  INFERX_RETURN_IF_ERROR(ValidateTensor(ctx, partial));
  if (ctx.Device() != device_) {
    return InvalidArgumentError("NCCL context must match the communicator's device");
  }
  if (partial.IsEmpty()) return OkStatus();
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  const NcclApi* api = NcclApi::Load();
  if (!api->Ready()) return Unavailable();
  return NcclCall(
      api->all_reduce(partial.Data(), partial.Data(), static_cast<size_t>(partial.Numel()),
                      kNcclBfloat16, kNcclSum, comm_, static_cast<void*>(ctx.GetStream())),
      "ncclAllReduce");
}

Status NcclComm::AllGatherLastDim(const ops::OpContext& ctx, const Tensor& partial,
                                  Tensor& full) {
  // NCCL's all-gather concatenates FLAT rank chunks, so [rows, shard] row
  // shards would land rank-major. Gathering per row keeps the result
  // [rows, size * shard] in the layout the caller expects.
  INFERX_RETURN_IF_ERROR(ValidateGather(ctx, partial, full));
  if (ctx.Device() != device_) {
    return InvalidArgumentError("NCCL context must match the communicator's device");
  }
  if (partial.IsEmpty()) return OkStatus();
  INFERX_RETURN_IF_ERROR(ctx.Runtime().Activate());
  const NcclApi* api = NcclApi::Load();
  if (!api->Ready()) return Unavailable();
  const int64_t shard = partial.Dim(1);
  const size_t elem = sizeof(uint16_t);
  const char* send = static_cast<const char*>(partial.Data());
  char* recv = static_cast<char*>(full.Data());
  for (int64_t row = 0; row < partial.Dim(0); ++row) {
    INFERX_RETURN_IF_ERROR(
        NcclCall(api->all_gather(send + row * shard * elem, recv + row * size() * shard * elem,
                                 static_cast<size_t>(shard), kNcclBfloat16, comm_,
                                 static_cast<void*>(ctx.GetStream())),
                 "ncclAllGather"));
  }
  return OkStatus();
}

struct NcclWorldState {
  enum class Kind { kSum, kGather };
  struct Submission {
    Kind kind;
    Tensor input;
    Tensor output;
    DeviceRuntime* runtime;
    Stream stream;
  };
  std::mutex mu;
  std::condition_variable cv;
  std::vector<DeviceId> devices;
  std::vector<void*> comms;
  std::vector<std::optional<Submission>> submissions;
  size_t arrived = 0;
  uint64_t generation = 0;
  Status failure;

  ~NcclWorldState() { AbortLocked(absl::CancelledError("NCCL world closed")); }

  // All NCCL API access (including health checks and teardown) is serialized.
  void AbortLocked(const Status& reason) {
    if (failure.ok())
      failure = reason.ok() ? absl::CancelledError("NCCL world aborted") : reason;
    const auto* api = NcclApi::Load();
    for (size_t rank = 0; rank < comms.size(); ++rank) {
      if (comms[rank] == nullptr) continue;
      if (auto runtime = RuntimeFor(devices[rank]); runtime.ok()) (void)(*runtime)->Activate();
      (void)api->comm_abort(comms[rank]);
      comms[rank] = nullptr;
    }
    cv.notify_all();
  }

  Status Dispatch() {
    const auto& first = *submissions.front();
    for (const auto& item : submissions) {
      if (item->kind != first.kind || item->input.GetShape() != first.input.GetShape()) {
        return InvalidArgumentError("NCCL ranks submitted different collectives or shapes");
      }
    }
    if (first.input.IsEmpty()) return OkStatus();
    const auto* api = NcclApi::Load();
    // One group per row preserves [rows, world * shard] gather layout.
    const int64_t rows = first.kind == Kind::kGather ? first.input.Dim(0) : 1;
    for (int64_t row = 0; row < rows; ++row) {
      INFERX_RETURN_IF_ERROR(NcclCall(api->group_start(), "ncclGroupStart"));
      Status status;
      for (size_t rank = 0; rank < comms.size(); ++rank) {
        const auto& item = *submissions[rank];
        status = item.runtime->Activate();
        if (!status.ok()) break;
        if (item.kind == Kind::kSum) {
          status = NcclCall(
              api->all_reduce(item.input.Data(), item.output.Data(), item.input.Numel(),
                              kNcclBfloat16, kNcclSum, comms[rank], item.stream.handle),
              "ncclAllReduce");
        } else {
          const int64_t shard = item.input.Dim(1);
          const auto* send = item.input.DataAs<uint16_t>() + row * shard;
          auto* recv = item.output.DataAs<uint16_t>() + row * shard * comms.size();
          status = NcclCall(api->all_gather(send, recv, shard, kNcclBfloat16, comms[rank],
                                            item.stream.handle),
                            "ncclAllGather");
        }
        if (!status.ok()) break;
      }
      const auto end = NcclCall(api->group_end(), "ncclGroupEnd");
      INFERX_RETURN_IF_ERROR(status);
      INFERX_RETURN_IF_ERROR(end);
    }
    return OkStatus();
  }

  Status Submit(int rank, Submission item) {
    std::unique_lock lock(mu);
    if (!failure.ok()) return failure;
    const auto current = generation;
    submissions[rank] = std::move(item);
    if (++arrived == comms.size()) {
      const Status status = Dispatch();
      if (!status.ok()) AbortLocked(status);
      arrived = 0;
      ++generation;
      cv.notify_all();
    } else {
      cv.wait(lock, [&] { return generation != current || !failure.ok(); });
    }
    return failure;
  }
};

namespace {
class NcclWorldRank final : public CommBackend {
 public:
  NcclWorldRank(std::shared_ptr<NcclWorldState> state, int rank)
      : state_(std::move(state)), rank_(rank) {}
  int size() const override { return state_->devices.size(); }
  int rank() const override { return rank_; }
  Status AllReduceSum(const ops::OpContext& ctx, Tensor& partial) override {
    INFERX_RETURN_IF_ERROR(Validate(ctx, ValidateTensor(ctx, partial)));
    return state_->Submit(
        rank_, {NcclWorldState::Kind::kSum, partial, partial, &ctx.Runtime(), ctx.GetStream()});
  }
  Status AllGatherLastDim(const ops::OpContext& ctx, const Tensor& partial,
                          Tensor& full) override {
    INFERX_RETURN_IF_ERROR(Validate(ctx, ValidateGather(ctx, partial, full)));
    return state_->Submit(
        rank_, {NcclWorldState::Kind::kGather, partial, full, &ctx.Runtime(), ctx.GetStream()});
  }

 private:
  Status Validate(const ops::OpContext& ctx, Status status) {
    if (status.ok() && ctx.Device() != state_->devices[rank_]) {
      status = InvalidArgumentError("NCCL rank context uses the wrong device");
    }
    if (!status.ok()) {
      std::lock_guard lock(state_->mu);
      state_->AbortLocked(status);
    }
    return status;
  }
  std::shared_ptr<NcclWorldState> state_;
  int rank_;
};
}  // namespace

NcclWorld::NcclWorld(std::shared_ptr<NcclWorldState> state) : state_(std::move(state)) {}

StatusOr<std::shared_ptr<NcclWorld>> NcclWorld::Create(const std::vector<DeviceId>& devices) {
  if (devices.empty()) return InvalidArgumentError("NCCL world needs at least one device");
  std::vector<int> ordinals;
  for (const auto device : devices) {
    if (!device.IsCuda()) return InvalidArgumentError("NCCL world requires CUDA devices");
    for (const auto ordinal : ordinals) {
      if (ordinal == device.index) return InvalidArgumentError("NCCL ranks need distinct GPUs");
    }
    INFERX_ASSIGN_OR_RETURN(auto runtime, RuntimeFor(device));
    INFERX_RETURN_IF_ERROR(runtime->Activate());
    ordinals.push_back(device.index);
  }
  const auto* api = NcclApi::Load();
  if (!api->Ready() || !api->comm_init_all || !api->comm_abort || !api->get_async_error ||
      !api->group_start || !api->group_end)
    return Unavailable();
  auto state = std::make_shared<NcclWorldState>();
  state->devices = devices;
  state->comms.resize(devices.size(), nullptr);
  state->submissions.resize(devices.size());
  INFERX_RETURN_IF_ERROR(
      NcclCall(api->comm_init_all(state->comms.data(), ordinals.size(), ordinals.data()),
               "ncclCommInitAll"));
  return std::shared_ptr<NcclWorld>(new NcclWorld(std::move(state)));
}

std::unique_ptr<CommBackend> NcclWorld::MakeRank(int rank) {
  return std::make_unique<NcclWorldRank>(state_, rank);
}

void NcclWorld::Abort(const Status& reason) {
  std::lock_guard lock(state_->mu);
  state_->AbortLocked(reason);
}

Status NcclWorld::CheckHealth() {
  std::lock_guard lock(state_->mu);
  if (!state_->failure.ok()) return state_->failure;
  const auto* api = NcclApi::Load();
  for (void* comm : state_->comms) {
    int error = kNcclSuccess;
    Status status = NcclCall(api->get_async_error(comm, &error), "ncclCommGetAsyncError");
    if (status.ok()) status = NcclCall(error, "NCCL asynchronous operation");
    if (!status.ok()) {
      state_->AbortLocked(status);
      return state_->failure;
    }
  }
  return OkStatus();
}

}  // namespace dist
}  // namespace inferx
