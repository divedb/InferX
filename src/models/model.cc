#include "inferx/models/model.h"

#include <utility>

namespace inferx {
namespace models {

namespace {

/// The binding map of a loadable model, or a precise complaint.
StatusOr<WeightBindable*> BindableOf(Model& model) {
  auto* bindable = dynamic_cast<WeightBindable*>(&model);
  if (bindable == nullptr) {
    return InternalError("model type ", typeid(model).name(),
                         " does not expose checkpoint bindings");
  }
  return bindable;
}

}  // namespace

Status LoadWeightMapInto(Model& model, WeightMap& weights,
                         CheckpointReader& checkpoint) {
  const Device device = model.GetModelSpec().device;
  INFERX_ASSIGN_OR_RETURN(DeviceRuntime* runtime, RuntimeFor(device));
  INFERX_RETURN_IF_ERROR(runtime->Activate());
  auto stream = runtime->Streams().CreateStream(StreamPriority::kNormal);
  INFERX_RETURN_IF_ERROR(stream.status());
  const Status loaded = LoadWeightsInto(checkpoint, weights, *runtime, stream->Get());
  const Status synced = runtime->Streams().SynchronizeStream(stream->Get());
  INFERX_RETURN_IF_ERROR(loaded);
  return synced;
}

Status LoadModelWeights(Model& model, CheckpointReader& checkpoint) {
  INFERX_ASSIGN_OR_RETURN(WeightBindable* bindable, BindableOf(model));
  WeightMap weights;
  INFERX_RETURN_IF_ERROR(bindable->BuildWeightMap(weights));
  return LoadWeightMapInto(model, weights, checkpoint);
}

Status FillModelWeightsWithZeros(Model& model) {
  INFERX_ASSIGN_OR_RETURN(WeightBindable* bindable, BindableOf(model));
  WeightMap weights;
  INFERX_RETURN_IF_ERROR(bindable->BuildWeightMap(weights));

  const Device device = model.GetModelSpec().device;
  INFERX_ASSIGN_OR_RETURN(DeviceRuntime* runtime, RuntimeFor(device));
  INFERX_RETURN_IF_ERROR(runtime->Activate());
  auto stream = runtime->Streams().CreateStream(StreamPriority::kNormal);
  INFERX_RETURN_IF_ERROR(stream.status());
  for (const auto& [name, binding] : weights.Bindings()) {
    (void)name;
    if (binding.dest != nullptr && !binding.dest->IsEmpty()) {
      INFERX_RETURN_IF_ERROR(runtime->Memory().MemsetAsync(
          binding.dest->Data(), uint8_t{0}, binding.dest->NBytes(), stream->Get()));
    }
  }
  return runtime->Streams().SynchronizeStream(stream->Get());
}

}  // namespace models
}  // namespace inferx
