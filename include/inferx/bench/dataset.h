#ifndef INFERX_BENCH_DATASET_H_
#define INFERX_BENCH_DATASET_H_

#include <cstdint>
#include <string>
#include <vector>

#include "inferx/core/status.h"

namespace inferx::bench {

/// \brief Prompt sources the benchmark commands can draw from.
enum class DatasetName : std::uint8_t {
  kRandom = 0,  ///< Synthetic fixed-length prompts.
};

/// \brief CLI spellings of DatasetName, for --dataset-name validation.
inline std::vector<std::string> DatasetNameValues() { return {"random"}; }

/// \brief Converts a validated CLI spelling to the typed value.
///
/// Only values from DatasetNameValues() are accepted.
inline StatusOr<DatasetName> ParseDatasetName(const std::string& name) {
  if (name == "random") return DatasetName::kRandom;
  return InvalidArgumentError("unknown dataset name: ", name);
}

/// \brief Prompt-source inputs shared by the benchmark entry points.
struct DatasetParams {
  DatasetName name = DatasetName::kRandom;
  std::string path;  ///< Dataset file; empty means the built-in behavior.
  int input_len = 32;
  int output_len = 128;
  int seed = 0;
};

}  // namespace inferx::bench

#endif  // INFERX_BENCH_DATASET_H_
