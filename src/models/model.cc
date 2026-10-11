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
  INFERX_ASSIGN_OR_RETURN(DeviceStream stream, runtime->CreateStream());
  const Status loaded = LoadWeightsInto(checkpoint, weights, *runtime, stream);
  const Status synced = runtime->SynchronizeStream(stream);
  const Status destroyed = runtime->DestroyStream(stream);
  INFERX_RETURN_IF_ERROR(loaded);
  INFERX_RETURN_IF_ERROR(synced);
  return destroyed;
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
  INFERX_ASSIGN_OR_RETURN(DeviceStream stream, runtime->CreateStream());
  for (const auto& [name, binding] : weights.Bindings()) {
    (void)name;
    if (binding.dest != nullptr && !binding.dest->IsEmpty()) {
      INFERX_RETURN_IF_ERROR(
          runtime->MemsetAsync(binding.dest->Data(), binding.dest->NBytes(), stream));
    }
  }
  INFERX_RETURN_IF_ERROR(runtime->SynchronizeStream(stream));
  return runtime->DestroyStream(stream);
}

}  // namespace models
}  // namespace inferx
