// Standalone profiling harness for GptOssModel::Forward.
//
// Loads the checkpoint, runs Forward over a short prompt with
// INFERX_GPTOSS_PROFILE set, and prints nothing else -- the model's own
// stderr report is the output. One run answers "where does the 24s go".
#include <cstdlib>
#include <iostream>
#include <vector>

#include "inferx/backends/cuda/cuda_utils.h"
#include "inferx/model/gpt_oss.h"

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "usage: " << argv[0] << " <gpt-oss-checkpoint-dir>\n";
    return 2;
  }
  if (!inferx::cuda::Available()) {
    std::cerr << "no CUDA device\n";
    return 1;
  }
  // The model reads this env var itself; set it so the caller does not have to.
  setenv("INFERX_GPTOSS_PROFILE", "1", 1);

  auto model =
      inferx::model::GptOssModel::Load(argv[1], inferx::DeviceId::Cuda(0));
  if (!model.ok()) {
    std::cerr << "load failed: " << model.status().ToString() << "\n";
    return 1;
  }

  // "The capital of France is" -- the 5-token reference prompt. Small enough
  // that attention is noise and the MoE upload dominates, which is the case
  // the profile exists to characterize.
  std::vector<int32_t> prompt = {976, 9029, 328, 10128, 382};
  std::vector<float> logits;
  auto s = model->Forward(prompt, &logits);
  if (!s.ok()) {
    std::cerr << "forward failed: " << s.ToString() << "\n";
    return 1;
  }
  return 0;
}
