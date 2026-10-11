#include "inferx/engine/model_executor.h"

#include <future>
#include <limits>
#include <utility>

#include "inferx/cache/kv_block_pool.h"
#include "inferx/core/logging.h"
#include "inferx/dist/nccl_comm.h"
#include "inferx/models/registry.h"
#include "inferx/util/hf_config.h"

namespace inferx {

struct ModelExecutor::Impl {
  /// Rank-local runners, ordered by communicator rank; ranks_[0] is the
  /// coordinating and (initially sole) sampling rank.
  std::vector<std::unique_ptr<models::ModelRunner>> ranks;
  std::function<void(const Status&)> abort;
  std::function<Status()> check_health;
  bool group = false;  ///< True when more than one rank participates.
};

ModelExecutor::ModelExecutor(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
ModelExecutor::~ModelExecutor() = default;

KvBlockPool* ModelExecutor::KvPool() { return impl_->ranks.front()->KvPool(); }
const models::ModelSpec& ModelExecutor::GetModelSpec() const {
  return impl_->ranks.front()->GetModelSpec();
}

StatusOr<std::unique_ptr<ModelExecutor>> ModelExecutor::Create(
    const ModelConfig& model, const CacheConfig& cache, const SchedulerConfig& scheduler,
    const ExecutionConfig& execution, const ParallelConfig& parallel,
    std::unique_ptr<CommBackend> comm) {
  INFERX_RETURN_IF_ERROR(parallel.Validate());
  INFERX_RETURN_IF_ERROR(model.device.Validate());
  if (comm || parallel.tensor_parallel_size == 1) {
    // Rank-local execution: one runner, with this rank's communicator.
    auto runner =
        models::ModelRunner::Create(model, cache, scheduler, execution, parallel, std::move(comm));
    INFERX_RETURN_IF_ERROR(runner.status());
    auto impl = std::make_unique<Impl>();
    impl->ranks.push_back(*std::move(runner));
    return std::unique_ptr<ModelExecutor>(new ModelExecutor(std::move(impl)));
  }

  // Multi-rank world construction: one rank-local runner per device.
  if (parallel.tensor_parallel_rank != 0) {
    return InvalidArgumentError("the tensor-parallel coordinator must have rank zero");
  }
  if (model.device.device_type != "cuda") {
    return InvalidArgumentError("multi-GPU tensor parallelism requires CUDA");
  }
  if (execution.enable_cuda_graphs) {
    return UnimplementedError("tensor-parallel serving currently requires eager execution");
  }
  std::vector<int> ids = model.device.device_ids;
  if (ids.empty()) {
    for (int rank = 0; rank < parallel.tensor_parallel_size; ++rank) ids.push_back(rank);
  }
  if (ids.size() != static_cast<size_t>(parallel.tensor_parallel_size)) {
    return InvalidArgumentError("device-ids must contain one device per tensor-parallel rank");
  }
  std::vector<Device> devices;
  for (const int id : ids) {
    if (id > std::numeric_limits<int8_t>::max()) {
      return InvalidArgumentError("CUDA device ordinal is out of range");
    }
    devices.push_back(Device::Cuda(static_cast<int8_t>(id)));
  }
  // Reject unknown architectures before creating the communication world.
  INFERX_ASSIGN_OR_RETURN(auto hf, HfConfig::FromFile(model.model_dir + "/config.json"));
  INFERX_RETURN_IF_ERROR(models::ArchitectureRegistry::Builtins()
                             .Resolve(hf.ModelType(), hf.Architecture())
                             .status());
  INFERX_ASSIGN_OR_RETURN(auto world, NcclWorld::Create(devices));
  std::vector<std::unique_ptr<models::ModelRunner>> ranks;
  for (int rank = 0; rank < parallel.tensor_parallel_size; ++rank) {
    ModelConfig local = model;
    local.device.device_ids = {ids[rank]};
    ParallelConfig rank_parallel;
    rank_parallel.tensor_parallel_size = parallel.tensor_parallel_size;
    rank_parallel.tensor_parallel_rank = rank;
    auto made = models::ModelRunner::Create(local, cache, scheduler, execution,
                                            rank_parallel, world->MakeRank(rank));
    if (!made.ok()) {
      world->Abort(made.status());
      return made.status();
    }
    ranks.push_back(*std::move(made));
  }
  return CreateGroup(std::move(ranks), [world](const Status& status) { world->Abort(status); },
                     [world] { return world->CheckHealth(); });
}

StatusOr<std::unique_ptr<ModelExecutor>> ModelExecutor::CreateGroup(
    std::vector<std::unique_ptr<models::ModelRunner>> ranks,
    std::function<void(const Status&)> abort, std::function<Status()> check_health) {
  if (ranks.empty() || !abort) {
    return InvalidArgumentError("rank group needs runners and abort");
  }
  for (size_t rank = 0; rank < ranks.size(); ++rank) {
    if (!ranks[rank]) return InvalidArgumentError("rank group contains a null runner");
    if (rank == 0) continue;
    if (ranks[rank]->GetModelSpec().vocab_size !=
        ranks.front()->GetModelSpec().vocab_size) {
      const auto status = InvalidArgumentError("rank group topology or capacities disagree");
      abort(status);
      return status;
    }
  }
  auto impl = std::make_unique<Impl>();
  impl->ranks = std::move(ranks);
  impl->abort = abort;
  impl->check_health = check_health;
  impl->group = impl->ranks.size() > 1;
  for (size_t rank = 0; rank < impl->ranks.size(); ++rank) {
    impl->ranks[rank]->SetFailureHooks(impl->abort, impl->check_health);
    if (rank != 0) impl->ranks[rank]->DisableSampling();
  }
  return std::unique_ptr<ModelExecutor>(new ModelExecutor(std::move(impl)));
}

StatusOr<ModelRunnerOutput> ModelExecutor::Execute(const SchedulerOutput& output) {
  if (!impl_->group) return impl_->ranks.front()->Run(output);

  // Dispatch the same logical step to every rank. Each rank's runner
  // validates and binds its rank-local state; a failure on any rank
  // poisons the group rather than retrying mutated state.
  std::vector<std::future<StatusOr<ModelRunnerOutput>>> pending;
  try {
    for (size_t rank = 1; rank < impl_->ranks.size(); ++rank) {
      pending.push_back(std::async(std::launch::async,
                                   [runner = impl_->ranks[rank].get(), &output] {
                                     return runner->Run(output);
                                   }));
    }
  } catch (const std::exception& error) {
    const Status failure = InternalError("could not start rank execution: ", error.what());
    impl_->abort(failure);
    return failure;  // Futures join before the borrowed output expires.
  }
  auto result = impl_->ranks.front()->Run(output);
  Status failure = result.status();
  for (auto& future : pending) {
    auto peer_result = future.get();
    if (failure.ok() && !peer_result.ok()) failure = peer_result.status();
  }
  if (!failure.ok()) {
    impl_->abort(failure);
    return failure;
  }
  // Only rank zero draws tokens. Share those exact ids with every rank,
  // including stochastic sampling, before admitting the next scheduler step.
  for (const auto& sample : result->samples) {
    if (sample.token_ids.empty()) continue;
    for (size_t rank = 1; rank < impl_->ranks.size(); ++rank) {
      INFERX_RETURN_IF_ERROR(impl_->ranks[rank]->ApplyAcceptedTokens(
          sample.request_id, sample.token_ids.back()));
    }
  }
  return result;
}

}  // namespace inferx
