#include "inferx/models/safetensors.h"

#include <filesystem>
#include <utility>

namespace inferx {
namespace models {

namespace {

/// Opens one shard, translating the reader's error type into a Status.
StatusOr<SafeTensorReader> OpenShard(const std::filesystem::path& path) {
  auto reader = SafeTensorReader::Open(path);
  if (!reader.has_value()) {
    const SafeTensorError error = std::move(reader.error());
    if (error.code == SafeTensorErrorCode::kFileNotFound) {
      return NotFoundError(error.message);
    }
    return InvalidArgumentError(error.message);
  }
  return std::move(*reader);
}

}  // namespace

StatusOr<CheckpointReader> CheckpointReader::Open(const std::string& dir) {
  const std::filesystem::path root(dir);
  std::error_code ec;
  if (!std::filesystem::is_directory(root, ec)) {
    return NotFoundError("not a directory: ", dir);
  }

  std::vector<SafeTensorReader> shards;
  const std::filesystem::path single = root / "model.safetensors";
  if (std::filesystem::exists(single)) {
    INFERX_ASSIGN_OR_RETURN(SafeTensorReader reader, OpenShard(single));
    shards.push_back(std::move(reader));
    return CheckpointReader(std::move(shards));
  }

  for (const auto& entry : std::filesystem::directory_iterator(root)) {
    if (!entry.is_regular_file() || entry.path().extension() != ".safetensors") {
      continue;
    }
    // Skip common auxiliary checkpoints.
    const std::string filename = entry.path().filename().string();
    if (filename.find("optim") != std::string::npos ||
        filename.find("adapter") != std::string::npos) {
      continue;
    }
    INFERX_ASSIGN_OR_RETURN(SafeTensorReader reader, OpenShard(entry.path()));
    shards.push_back(std::move(reader));
  }
  if (shards.empty()) {
    return NotFoundError("no .safetensors files found in ", dir);
  }
  return CheckpointReader(std::move(shards));
}

StatusOr<Tensor> CheckpointReader::GetTensor(const std::string& name) const {
  for (const SafeTensorReader& shard : shards_) {
    if (std::optional<Tensor> tensor = shard.GetTensor(name)) {
      return *std::move(tensor);
    }
  }
  return NotFoundError("checkpoint has no tensor named ", name);
}

std::vector<std::string> CheckpointReader::Names() const {
  std::vector<std::string> names;
  for (const SafeTensorReader& shard : shards_) {
    for (const std::string& name : shard.TensorNames()) names.push_back(name);
  }
  return names;
}

}  // namespace models
}  // namespace inferx
