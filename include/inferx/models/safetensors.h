#ifndef INFERX_SAFETENSORS_H_
#define INFERX_SAFETENSORS_H_

#include <cstdint>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "inferx/core/tensor.h"

namespace inferx {

/// \brief A single tensor converted to float32, with its original shape.
struct SafeTensor {
  std::vector<int64_t> shape;  ///< Original tensor shape.
  std::vector<float> data;     ///< Element values in row-major order.
};

/// \brief A collection of tensors, possibly merged from multiple shards.
class SafeTensors {
 public:
  /// \brief Loads a single `.safetensors` file.
  ///
  /// \param path Path to the `.safetensors` file.
  /// \return The loaded tensors, or an error status.
  static absl::StatusOr<SafeTensors> FromFile(const std::string& path);
  /// \brief Loads `model.safetensors` from a directory, or merges all shards
  /// of a sharded checkpoint.
  ///
  /// \param dir Model directory containing the checkpoint.
  /// \return The loaded tensors, or an error status.
  static absl::StatusOr<SafeTensors> FromDirectory(const std::string& dir);

  /// \brief Checks whether a tensor with the given name exists.
  ///
  /// \param name Tensor name.
  /// \return True if the tensor is present.
  bool Contains(absl::string_view name) const;
  /// \brief Finds a tensor by name.
  ///
  /// \param name Tensor name.
  /// \return Pointer to the tensor, or nullptr if not present.
  const SafeTensor* Find(absl::string_view name) const;
  /// \brief Returns the number of stored tensors.
  int64_t size() const { return static_cast<int64_t>(tensors_.size()); }

  // Loads f32 data onto device. Transposition is supported for rank-2 weights.
  StatusOr<Tensor> GetTensor(absl::string_view name, DeviceId device,
                             bool transpose = false) const;
  // Row slice of a rank-2 tensor, non-transposed: rows [begin, end).
  StatusOr<Tensor> GetTensorRows(absl::string_view name, DeviceId device,
                                 int64_t begin, int64_t end) const;
  // Column-parallel shard: rows [out_begin, out_end) of the checkpoint's
  // [out, in] matrix, transposed to [in, out_end - out_begin].
  StatusOr<Tensor> GetTransposedOutputSlice(absl::string_view name,
                                            DeviceId device, int64_t out_begin,
                                            int64_t out_end) const;
  // Row-parallel shard: input columns [in_begin, in_end) of the checkpoint's
  // [out, in] matrix, returned transposed as [in_end - in_begin, out].
  StatusOr<Tensor> GetTransposedInputSlice(absl::string_view name,
                                           DeviceId device, int64_t in_begin,
                                           int64_t in_end) const;

 private:
  /// \brief Moves every tensor of `other` into this collection.
  ///
  /// \param other Collection whose entries are merged in.
  void Merge(SafeTensors&& other);

  absl::flat_hash_map<std::string, SafeTensor> tensors_;
};

}  // namespace inferx

#endif  // INFERX_SAFETENSORS_H_
