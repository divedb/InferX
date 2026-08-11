#include "qwen_runner.h"

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include "inferx/comm/nccl_communicator.h"
#include "inferx/core/device_runtime.h"
#include "kv_autosize.h"

namespace inferx::engine {
namespace {

int64_t SteadyNowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

double ProgressAgeSeconds(int64_t last_progress_ns) {
  if (last_progress_ns == 0) return 0.0;
  return static_cast<double>(SteadyNowNs() - last_progress_ns) / 1e9;
}

/// TP ranks auto-size their KV pool independently but must attach identical
/// block counts -- the scheduler runs one block table for all ranks -- so
/// each posts its local answer and everyone proceeds with the minimum. A rank
/// that fails posts a failure sentinel instead, so its peer errors out rather
/// than waiting forever.
class KvSizeRendezvous {
 public:
  explicit KvSizeRendezvous(int expected) : expected_(expected) {}

  /// Posts `local_blocks` (<= 0 marks failure) and returns the agreed
  /// minimum, or a non-positive value when any rank failed or the peer never
  /// arrived.
  int64_t Agree(int64_t local_blocks) {
    std::unique_lock lock(mu_);
    min_blocks_ = std::min(min_blocks_, local_blocks);
    ++arrived_;
    cv_.notify_all();
    if (!cv_.wait_for(lock, std::chrono::minutes(5),
                      [this] { return arrived_ >= expected_; })) {
      return -1;
    }
    return min_blocks_;
  }

 private:
  const int expected_;
  std::mutex mu_;
  std::condition_variable cv_;
  int arrived_ = 0;
  int64_t min_blocks_ = std::numeric_limits<int64_t>::max();
};

/// `agreed` reports whether this rank posted to `rendezvous`, so failure
/// paths know whether the peer still needs unblocking.
Status ConfigureModel(model::Qwen2Model* model, const QwenRunnerConfig& config,
                      DeviceId device, KvSizeRendezvous* rendezvous,
                      bool* agreed) {
  Status prep = OkStatus();
  if (config.fp8_weights) prep = model->QuantizeWeightsToF8();
  if (prep.ok() && config.int4_weights) prep = model->QuantizeWeightsToInt4();
  if (prep.ok() && config.fp8_kv_cache) prep = model->EnableFp8KvCache();

  int64_t blocks = config.kv_blocks;
  if (blocks <= 0) {
    StatusOr<int64_t> sized = prep;
    if (prep.ok()) {
      KvSizingSpec spec;
      spec.explicit_bytes = config.kv_cache_memory_bytes;
      spec.gpu_memory_utilization = config.gpu_memory_utilization;
      spec.block_bytes = model->KvBlockBytes(config.block_size);
      spec.min_blocks =
          (config.max_seq_len + config.block_size - 1) / config.block_size;
      sized = AutosizeKvBlocksOnDevice(spec, device);
    }
    if (rendezvous != nullptr) {
      const int64_t agreed_blocks = rendezvous->Agree(sized.ok() ? *sized : -1);
      *agreed = true;
      INFERX_RETURN_IF_ERROR(sized.status());
      if (agreed_blocks <= 0) {
        return InternalError("peer rank failed during KV cache sizing");
      }
      blocks = agreed_blocks;
    } else {
      INFERX_RETURN_IF_ERROR(sized.status());
      blocks = *sized;
    }
  } else {
    INFERX_RETURN_IF_ERROR(prep);
  }
  INFERX_RETURN_IF_ERROR(prep);
  INFERX_RETURN_IF_ERROR(model->AttachKvCache(blocks, config.block_size));
  return model->EnableDeviceSampling(config.max_sampling_rows);
}

class SingleQwenRunner final : public QwenRunner {
 public:
  SingleQwenRunner(model::Qwen2Model model, int device,
                   std::shared_ptr<comm::CommMetrics> communication)
      : model_(std::move(model)),
        device_(device),
        communication_(std::move(communication)) {}

  KvBlockPool* kv_pool() override { return model_.kv_pool(); }
  Status ReserveActivations(int64_t n) override {
    return model_.ReserveActivations(n);
  }
  Status StepAsync(const model::ForwardBatch& batch) override {
    Status result = model_.StepAsync(batch);
    if (!result.ok()) healthy_.store(false, std::memory_order_relaxed);
    return result;
  }
  Status AwaitStep(std::vector<int32_t>* sampled) override {
    Status result = model_.AwaitStep(sampled);
    healthy_.store(result.ok(), std::memory_order_relaxed);
    if (result.ok()) {
      last_step_device_ms_.store(model_.last_step_device_ms(),
                                 std::memory_order_relaxed);
      last_progress_ns_.store(SteadyNowNs(), std::memory_order_relaxed);
    }
    return result;
  }
  Status CaptureDecodeGraph(int64_t n, int64_t blocks) override {
    return model_.CaptureDecodeGraph(n, blocks);
  }
  Status ReadSampledLogprobs(
      std::vector<model::Qwen2Model::SampledLogprob>* out) override {
    return model_.ReadSampledLogprobs(out);
  }
  float last_step_device_ms() const override {
    return last_step_device_ms_.load(std::memory_order_relaxed);
  }
  std::vector<RankTelemetry> telemetry() const override {
    return {{.rank = 0,
             .device = device_,
             .world_size = 1,
             .backend = comm::CommBackend::kSingleRank,
             .healthy = healthy_.load(std::memory_order_relaxed),
             .last_progress_age_seconds = ProgressAgeSeconds(
                 last_progress_ns_.load(std::memory_order_relaxed)),
             .last_step_device_ms =
                 last_step_device_ms_.load(std::memory_order_relaxed),
             .timeouts = 0,
             .communication = communication_->Snapshot()}};
  }

 private:
  model::Qwen2Model model_;
  int device_ = 0;
  std::shared_ptr<comm::CommMetrics> communication_;
  std::atomic<bool> healthy_{true};
  std::atomic<int64_t> last_progress_ns_{0};
  std::atomic<float> last_step_device_ms_{0.0f};
};

class RankWorker {
 public:
  using Job = std::function<Status(model::Qwen2Model&)>;

  RankWorker(const QwenRunnerConfig& config, int rank,
             const comm::NcclUniqueIdBytes& unique_id,
             std::shared_ptr<KvSizeRendezvous> rendezvous)
      : config_(config),
        rank_(rank),
        unique_id_(unique_id),
        rendezvous_(std::move(rendezvous)),
        communication_(std::make_shared<comm::CommMetrics>()) {
    thread_ = std::thread([this] { ThreadMain(); });
  }

  ~RankWorker() { Stop(); }

  Status WaitInitialized() { return initialized_.get_future().get(); }

  Status Submit(Job job) {
    std::lock_guard lock(mu_);
    if (stopping_) return FailedPreconditionError("rank worker is stopping");
    if (has_job_)
      return FailedPreconditionError("rank worker already has work");
    job_ = std::move(job);
    has_job_ = true;
    done_ = false;
    cv_.notify_one();
    return OkStatus();
  }

  Status Wait() {
    std::unique_lock lock(mu_);
    cv_.wait(lock, [this] { return done_ || stopping_; });
    return result_;
  }

  bool WaitFor(std::chrono::seconds timeout, Status* result) {
    std::unique_lock lock(mu_);
    if (!cv_.wait_for(lock, timeout, [this] { return done_ || stopping_; })) {
      return false;
    }
    *result = result_;
    return true;
  }

  Status AbortCommunication() {
    return model_ == nullptr ? OkStatus() : model_->AbortCommunicator();
  }

  KvBlockPool* kv_pool() const { return pool_; }
  float last_step_device_ms() const { return last_step_device_ms_; }

  RankTelemetry telemetry() const {
    return {.rank = rank_,
            .device = config_.devices[static_cast<size_t>(rank_)],
            .world_size = static_cast<int>(config_.devices.size()),
            .backend = comm::CommBackend::kNccl,
            .healthy = healthy_.load(std::memory_order_relaxed),
            .last_progress_age_seconds = ProgressAgeSeconds(
                last_progress_ns_.load(std::memory_order_relaxed)),
            .last_step_device_ms =
                last_step_device_ms_atomic_.load(std::memory_order_relaxed),
            .timeouts = timeouts_.load(std::memory_order_relaxed),
            .communication = communication_->Snapshot()};
  }

  void RecordTimeout() {
    timeouts_.fetch_add(1, std::memory_order_relaxed);
    healthy_.store(false, std::memory_order_relaxed);
  }

  void Stop() {
    {
      std::lock_guard lock(mu_);
      if (stopping_) return;
      stopping_ = true;
      cv_.notify_all();
    }
    if (thread_.joinable()) thread_.join();
  }

 private:
  void ThreadMain() {
    const int device = config_.devices[static_cast<size_t>(rank_)];
    auto runtime = RuntimeFor(DeviceId::Cuda(static_cast<int8_t>(device)));
    Status status =
        runtime.ok()
            ? (*runtime)->SetDevice(DeviceId::Cuda(static_cast<int8_t>(device)))
            : runtime.status();
    if (status.ok()) {
      comm::NcclCommConfig comm_config;
      comm_config.rank = rank_;
      comm_config.world_size = static_cast<int>(config_.devices.size());
      comm_config.local_device = device;
      comm_config.unique_id = unique_id_;
      auto communicator = comm::CreateNcclCommunicator(comm_config);
      if (!communicator.ok()) {
        status = communicator.status();
      } else {
        *communicator = comm::ObserveCommunicator(
            std::move(*communicator), communication_,
            {.timing_sample_every = config_.collective_timing_sample_every});
        auto loaded = model::Qwen2Model::LoadFromDirectory(
            config_.model_dir, std::move(*communicator));
        if (!loaded.ok()) {
          status = loaded.status();
        } else {
          model_ = std::make_unique<model::Qwen2Model>(std::move(*loaded));
          status = ConfigureModel(model_.get(), config_,
                                  DeviceId::Cuda(static_cast<int8_t>(device)),
                                  rendezvous_.get(), &agreed_with_peer_);
          if (status.ok()) pool_ = model_->kv_pool();
        }
      }
    }
    // A rank that died before posting its KV size would leave the peer
    // blocked in Agree until its timeout; post the failure sentinel instead.
    if (!status.ok() && rendezvous_ != nullptr && !agreed_with_peer_) {
      (void)rendezvous_->Agree(-1);
      agreed_with_peer_ = true;
    }
    healthy_.store(status.ok(), std::memory_order_relaxed);
    initialized_.set_value(status);

    while (true) {
      Job job;
      {
        std::unique_lock lock(mu_);
        cv_.wait(lock, [this] { return has_job_ || stopping_; });
        if (stopping_) break;
        job = std::move(job_);
        has_job_ = false;
      }

      Status result = status.ok() ? job(*model_) : status;
      if (model_ != nullptr) {
        last_step_device_ms_ = model_->last_step_device_ms();
        last_step_device_ms_atomic_.store(last_step_device_ms_,
                                          std::memory_order_relaxed);
      }
      healthy_.store(result.ok(), std::memory_order_relaxed);
      if (result.ok()) {
        last_progress_ns_.store(SteadyNowNs(), std::memory_order_relaxed);
      }
      {
        std::lock_guard lock(mu_);
        result_ = std::move(result);
        done_ = true;
        cv_.notify_all();
      }
    }

    model_.reset();
  }

  QwenRunnerConfig config_;
  int rank_;
  comm::NcclUniqueIdBytes unique_id_;
  std::shared_ptr<KvSizeRendezvous> rendezvous_;
  bool agreed_with_peer_ = false;
  std::thread thread_;
  std::promise<Status> initialized_;
  std::unique_ptr<model::Qwen2Model> model_;
  KvBlockPool* pool_ = nullptr;
  float last_step_device_ms_ = 0.0f;
  std::shared_ptr<comm::CommMetrics> communication_;
  std::atomic<bool> healthy_{false};
  std::atomic<int64_t> last_progress_ns_{0};
  std::atomic<float> last_step_device_ms_atomic_{0.0f};
  std::atomic<uint64_t> timeouts_{0};

  mutable std::mutex mu_;
  std::condition_variable cv_;
  Job job_;
  Status result_;
  bool has_job_ = false;
  bool done_ = false;
  bool stopping_ = false;
};

class NcclQwenRunner final : public QwenRunner {
 public:
  static StatusOr<std::unique_ptr<NcclQwenRunner>> Create(
      const QwenRunnerConfig& config) {
    if (config.devices.size() != 2) {
      return InvalidArgumentError(
          "the first NCCL runtime requires exactly two devices, got ",
          config.devices.size());
    }
    if (config.devices[0] == config.devices[1]) {
      return InvalidArgumentError("NCCL ranks require distinct CUDA devices");
    }
    INFERX_ASSIGN_OR_RETURN(const comm::NcclUniqueIdBytes id,
                            comm::CreateNcclUniqueId());

    auto runner = std::unique_ptr<NcclQwenRunner>(new NcclQwenRunner());
    auto rendezvous = std::make_shared<KvSizeRendezvous>(2);
    for (int rank = 0; rank < 2; ++rank) {
      runner->workers_.push_back(
          std::make_unique<RankWorker>(config, rank, id, rendezvous));
    }
    for (auto& worker : runner->workers_) {
      const Status initialized = worker->WaitInitialized();
      if (!initialized.ok()) return initialized;
    }
    return runner;
  }

  KvBlockPool* kv_pool() override { return workers_[0]->kv_pool(); }

  Status ReserveActivations(int64_t max_tokens) override {
    return Dispatch([max_tokens](model::Qwen2Model& model) {
      return model.ReserveActivations(max_tokens);
    });
  }

  Status StepAsync(const model::ForwardBatch& batch) override {
    return Dispatch(
        [batch](model::Qwen2Model& model) { return model.StepAsync(batch); });
  }

  Status AwaitStep(std::vector<int32_t>* sampled) override {
    std::vector<std::vector<int32_t>> outputs(workers_.size());
    INFERX_RETURN_IF_ERROR(
        DispatchIndexed([&outputs](size_t rank, model::Qwen2Model& model) {
          return model.AwaitStep(&outputs[rank]);
        }));
    *sampled = std::move(outputs[0]);
    return OkStatus();
  }

  Status CaptureDecodeGraph(int64_t num_seqs,
                            int64_t max_blocks_per_seq) override {
    return Dispatch([=](model::Qwen2Model& model) {
      return model.CaptureDecodeGraph(num_seqs, max_blocks_per_seq);
    });
  }

  Status ReadSampledLogprobs(
      std::vector<model::Qwen2Model::SampledLogprob>* out) override {
    // Rank 0 only: every rank samples the same tokens from the same
    // all-reduced logits, so one readback answers for the pair.
    INFERX_RETURN_IF_ERROR(workers_[0]->Submit([out](model::Qwen2Model& model) {
      return model.ReadSampledLogprobs(out);
    }));
    return workers_[0]->Wait();
  }

  float last_step_device_ms() const override {
    return workers_[0]->last_step_device_ms();
  }

  std::vector<RankTelemetry> telemetry() const override {
    std::vector<RankTelemetry> result;
    result.reserve(workers_.size());
    for (const auto& worker : workers_) result.push_back(worker->telemetry());
    return result;
  }

 private:
  template <typename Fn>
  Status Dispatch(Fn fn) {
    return DispatchIndexed(
        [fn = std::move(fn)](size_t, model::Qwen2Model& model) mutable {
          return fn(model);
        });
  }

  template <typename Fn>
  Status DispatchIndexed(Fn fn) {
    for (size_t rank = 0; rank < workers_.size(); ++rank) {
      INFERX_RETURN_IF_ERROR(
          workers_[rank]->Submit([fn, rank](model::Qwen2Model& model) mutable {
            return fn(rank, model);
          }));
    }
    Status first = OkStatus();
    for (auto& worker : workers_) {
      Status status;
      if (!worker->WaitFor(std::chrono::seconds(30), &status)) {
        worker->RecordTimeout();
        for (auto& peer : workers_) (void)peer->AbortCommunication();
        return InternalError(
            "tensor-parallel rank timed out; all communicators were aborted");
      }
      if (first.ok() && !status.ok()) first = status;
    }
    if (!first.ok()) {
      for (auto& worker : workers_) (void)worker->AbortCommunication();
    }
    return first;
  }

  std::vector<std::unique_ptr<RankWorker>> workers_;
};

}  // namespace

StatusOr<std::unique_ptr<QwenRunner>> QwenRunner::Create(
    const QwenRunnerConfig& config) {
  if (config.devices.empty()) {
    return InvalidArgumentError("at least one CUDA device is required");
  }
  if (config.fp8_weights && config.int4_weights) {
    return InvalidArgumentError("FP8 and int4 weights are mutually exclusive");
  }
  if (config.use_nccl) {
    INFERX_ASSIGN_OR_RETURN(auto runner, NcclQwenRunner::Create(config));
    return std::unique_ptr<QwenRunner>(std::move(runner));
  }
  if (config.devices.size() != 1) {
    return InvalidArgumentError("multiple devices require the NCCL backend");
  }

  const int device = config.devices.front();
  INFERX_ASSIGN_OR_RETURN(
      DeviceRuntime * runtime,
      RuntimeFor(DeviceId::Cuda(static_cast<int8_t>(device))));
  INFERX_RETURN_IF_ERROR(
      runtime->SetDevice(DeviceId::Cuda(static_cast<int8_t>(device))));
  auto communicator = std::make_unique<comm::SingleRankComm>(
      DeviceId::Cuda(static_cast<int8_t>(device)));
  auto communication = std::make_shared<comm::CommMetrics>();
  std::unique_ptr<comm::Communicator> observed = comm::ObserveCommunicator(
      std::move(communicator), communication,
      {.timing_sample_every = config.collective_timing_sample_every});
  INFERX_ASSIGN_OR_RETURN(model::Qwen2Model model,
                          model::Qwen2Model::LoadFromDirectory(
                              config.model_dir, std::move(observed)));
  bool agreed = false;
  INFERX_RETURN_IF_ERROR(ConfigureModel(
      &model, config, DeviceId::Cuda(static_cast<int8_t>(device)), nullptr,
      &agreed));
  return std::unique_ptr<QwenRunner>(std::make_unique<SingleQwenRunner>(
      std::move(model), device, std::move(communication)));
}

}  // namespace inferx::engine
