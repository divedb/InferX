#include <gtest/gtest.h>

#include "engine/model_runner_validation.h"

namespace inferx::engine {
namespace {

TEST(ModelRunnerValidationTest, QwenFamilyAcceptsConfiguredFeatures) {
  const ModelRunnerFeatures features{.tensor_parallel_size = 2,
                                     .fp8_weights = true,
                                     .int4_weights = false,
                                     .fp8_kv_cache = true};
  EXPECT_TRUE(
      ValidateModelRunnerFeatures(model::Architecture::kQwen2, features).ok());
  EXPECT_TRUE(ValidateModelRunnerFeatures(model::Architecture::kQwen2Moe,
                                          features)
                  .ok());
  EXPECT_TRUE(
      ValidateModelRunnerFeatures(model::Architecture::kLlama, features).ok());
}

TEST(ModelRunnerValidationTest, SynchronousArchitecturesRejectTensorParallel) {
  const ModelRunnerFeatures features{.tensor_parallel_size = 2};
  const Status deepseek =
      ValidateModelRunnerFeatures(model::Architecture::kDeepSeekV2, features);
  const Status gpt =
      ValidateModelRunnerFeatures(model::Architecture::kGptOss, features);

  EXPECT_EQ(deepseek.code(), absl::StatusCode::kUnimplemented);
  EXPECT_EQ(gpt.code(), absl::StatusCode::kUnimplemented);
}

TEST(ModelRunnerValidationTest, SynchronousArchitecturesRejectQuantization) {
  for (const model::Architecture architecture :
       {model::Architecture::kDeepSeekV2, model::Architecture::kGptOss}) {
    EXPECT_EQ(ValidateModelRunnerFeatures(architecture, {.fp8_weights = true})
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ValidateModelRunnerFeatures(architecture, {.int4_weights = true})
                  .code(),
              absl::StatusCode::kInvalidArgument);
    EXPECT_EQ(ValidateModelRunnerFeatures(architecture, {.fp8_kv_cache = true})
                  .code(),
              absl::StatusCode::kInvalidArgument);
  }
}

TEST(ModelRunnerValidationTest, SynchronousDefaultsAreSupported) {
  EXPECT_TRUE(ValidateModelRunnerFeatures(model::Architecture::kDeepSeekV2, {})
                  .ok());
  EXPECT_TRUE(
      ValidateModelRunnerFeatures(model::Architecture::kGptOss, {}).ok());
}

}  // namespace
}  // namespace inferx::engine
