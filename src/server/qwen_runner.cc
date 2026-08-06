#include "qwen_runner.h"

#include <condition_variable>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <thread>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

#include "inferx/comm/nccl_communicator.h"
#include "inferx/core/cuda_utils.h"

namespace inferx::server {
namespace {

Status ConfigureModel(model::Qwen2Model* model,
                      const QwenRunnerConfig& config) {
  if (config.fp8_weights) {
    INFERX_RETURN_IF_ERROR(model->QuantizeWeightsToF8());
  }
  if (config.int4_weights) {
    INFERX_RETURN_IF_ERROR(model->QuantizeWeightsToInt4());
  }
  if (config.fp8_kv_cache) {
    INFERX_RETURN_IF_ERROR(model->EnableFp8KvCache());
  }
  INFERX_RETURN_IF_ERROR(
      model->AttachKvCache(config.kv_blocks, config.block_size));
  return model->EnableDeviceSampling(config.max_sampling_rows);
}

class SingleQwenRunner final : public QwenRunner {
 public:
  explicit SingleQwenRunner(model::Qwen2Model model)
      : model_(std::move(model)) {}

  KvBlockPool* kv_pool() override { return model_.kv_pool(); }
  Status ReserveActivations(int64_t n) override {
    return model_.ReserveActivations(n);
  }
  Status StepAsync(const model::ForwardBatch& batch) override {
    return model_.StepAsync(batch);
  }
  Status AwaitStep(std::vector<int32_t>* sampled) override {
    return model_.AwaitStep(sampled);
  }
  Status CaptureDecodeGraph(int64_t n, int64_t blocks) override {
    return model_.CaptureDecodeGraph(n, blocks);
  }
  float last_step_device_ms() const override {
    return model_.last_step_device_ms();
  }

 private:
  model::Qwen2Model model_;
};

class RankWorker {
 public:
  using Job = std::function<Status(model::Qwen2Model&)>;

  RankWorker(const QwenRunnerConfig& config, int rank,
             const comm::NcclUniqueIdBytes& unique_id)
      : config_(config), rank_(rank), unique_id_(unique_id) {
    thread_ = std::thread([this] { ThreadMain(); });
  }

  ~RankWorker() { Stop(); }

  Status WaitInitialized() { return initialized_.get_future().get(); }

  Status Submit(Job job) {
    std::lock_guard lock(mu_);
    if (stopping_) return FailedPreconditionError("rank worker is stopping");
    if (has_job_) return FailedPreconditionError("rank worker already has work");
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

  KvBlockPool* kv_pool() const { return pool_; }
  float last_step_device_ms() const { return last_step_device_ms_; }

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
    Status status = CudaErrorToStatus(cudaSetDevice(device), "cudaSetDevice",
                                     __FILE__, __LINE__);
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
        auto loaded = model::Qwen2Model::LoadFromDirectory(
            config_.model_dir, std::move(*communicator));
        if (!loaded.ok()) {
          status = loaded.status();
        } else {
          model_ = std::make_unique<model::Qwen2Model>(std::move(*loaded));
          status = ConfigureModel(model_.get(), config_);
          if (status.ok()) pool_ = model_->kv_pool();
        }
      }
    }
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
      if (model_ != nullptr) last_step_device_ms_ = model_->last_step_device_ms();
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
  std::thread thread_;
  std::promise<Status> initialized_;
  std::unique_ptr<model::Qwen2Model> model_;
  KvBlockPool* pool_ = nullptr;
  float last_step_device_ms_ = 0.0f;

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
    for (int rank = 0; rank < 2; ++rank) {
      runner->workers_.push_back(
          std::make_unique<RankWorker>(config, rank, id));
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
    return Dispatch([batch](model::Qwen2Model& model) {
      return model.StepAsync(batch);
    });
  }

  Status AwaitStep(std::vector<int32_t>* sampled) override {
    std::vector<std::vector<int32_t>> outputs(workers_.size());
    INFERX_RETURN_IF_ERROR(DispatchIndexed(
        [&outputs](size_t rank, model::Qwen2Model& model) {
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

  float last_step_device_ms() const override {
    return workers_[0]->last_step_device_ms();
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
      INFERX_RETURN_IF_ERROR(workers_[rank]->Submit(
          [fn, rank](model::Qwen2Model& model) mutable {
            return fn(rank, model);
          }));
    }
    Status first = OkStatus();
    for (auto& worker : workers_) {
      const Status status = worker->Wait();
      if (first.ok() && !status.ok()) first = status;
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
  INFERX_CUDA_RETURN_IF_ERROR(cudaSetDevice(device));
  auto communicator = std::make_unique<comm::SingleRankComm>(
      DeviceId::Cuda(static_cast<int8_t>(device)));
  INFERX_ASSIGN_OR_RETURN(
      model::Qwen2Model model,
      model::Qwen2Model::LoadFromDirectory(config.model_dir,
                                           std::move(communicator)));
  INFERX_RETURN_IF_ERROR(ConfigureModel(&model, config));
  return std::unique_ptr<QwenRunner>(
      std::make_unique<SingleQwenRunner>(std::move(model)));
}

}  // namespace inferx::server
