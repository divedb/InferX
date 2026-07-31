#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <memory>

#include "inferx/core/device_buffer.h"
#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx::kernels {

/// \brief A cuBLASLt context with a per-shape plan cache.
///
/// Owns the library handle, one workspace allocation, and a cache of matmul
/// plans keyed on `(m, n, k)`. Move-only, and **not thread-safe** -- cuBLASLt
/// handles are not, and the cache is a plain hash map. The intent is one
/// instance per executor rank thread, constructed at startup.
///
/// Every plan is built on first use and reused thereafter. `Warm()` exists so
/// that first use can be moved off the step path deliberately rather than
/// happening inside whichever request arrives first.
class CublasLtGemm {
 public:
  /// The cuBLASLt workspace. 32 MiB is NVIDIA's guidance for Ada-class parts;
  /// too small and the heuristic silently declines split-k algorithms that need
  /// scratch, which shows up as a mysteriously slow GEMM rather than an error.
  static constexpr size_t kDefaultWorkspaceBytes = 32u << 20;

  /// \brief Creates a context on the current CUDA device.
  ///
  /// \param workspace_bytes Size of the algorithm workspace.
  /// \return                The context, or the cuBLAS/CUDA error.
  static StatusOr<CublasLtGemm> Create(
      size_t workspace_bytes = kDefaultWorkspaceBytes);

  ~CublasLtGemm();

  CublasLtGemm(const CublasLtGemm&) = delete;
  CublasLtGemm& operator=(const CublasLtGemm&) = delete;
  CublasLtGemm(CublasLtGemm&&) noexcept;
  CublasLtGemm& operator=(CublasLtGemm&&) noexcept;

  /// \brief Computes `y = x * wᵀ` in FP16 with FP32 accumulation.
  ///
  /// The shapes are `nn.Linear`'s, not textbook GEMM's: the weight is stored
  /// `[out, in]` because that is how every checkpoint on disk stores it, and
  /// transposing it at load time to suit a GEMM convention would cost a copy of
  /// the whole model. Both operands are therefore K-major, which is also the
  /// layout the tensor cores want.
  ///
  /// \param x      Activations, `[m, k]` f16, row-major, on a CUDA device.
  /// \param w      Weights, `[n, k]` f16, row-major, on the same device.
  /// \param y      Output, `[m, n]` f16, row-major, on the same device.
  /// \param stream Stream to launch on. Does **not** synchronize.
  /// \return       OK, InvalidArgument for shapes/dtypes this cannot serve, or
  ///               the cuBLAS error.
  Status LinearF16(const TensorView& x, const TensorView& w,
                   const TensorView& y, cudaStream_t stream = nullptr);

  /// \brief `y = x · wᵀ` in bf16 with FP32 accumulation.
  ///
  /// The path the model layer runs on, because bf16 is what checkpoints store.
  /// `LinearF16` remains as the M1 benchmark baseline; the two are separate
  /// entry points rather than a dtype switch so that neither can be reached by
  /// accident from the other's caller.
  ///
  /// \param x      Activations, `[m, k]` bf16, row-major, on a CUDA device.
  /// \param w      Weights, `[n, k]` bf16, row-major, same device.
  /// \param y      Output, `[m, n]` bf16, row-major.
  /// \param stream Stream to launch on. Does **not** synchronize.
  Status LinearBF16(const TensorView& x, const TensorView& w,
                    const TensorView& y, cudaStream_t stream = nullptr);

  /// \brief Computes `y = (x·x_scale) · (w·w_scale)ᵀ` in FP8 e4m3.
  ///
  /// The M1 prototype (ARCHITECTURE.md §15 Q2). Same shapes and same layout
  /// contract as `LinearF16` -- which is not a coincidence: cuBLASLt's FP8 path
  /// *requires* the TN layout that the FP16 mapping already produces, so the
  /// weights being stored `[out, in]` is what makes both work unchanged.
  ///
  /// Accumulation is FP32. The scales are device pointers because they are
  /// computed on the device by `ComputeF8Scale` and must not round-trip to the
  /// host on a serving path; cuBLASLt reads them at launch and folds them into
  /// the epilogue, so the result lands in the original units with no separate
  /// dequantization pass.
  ///
  /// \param x           Activations, `[m, k]` f8e4m3, row-major, on a device.
  /// \param w           Weights, `[n, k]` f8e4m3, row-major, same device.
  /// \param y           Output, `[m, n]` **f16** -- FP8 out would need a third
  ///                    scale and lose precision the epilogue already has.
  /// \param x_scale_dev Device pointer to x's dequant scale.
  /// \param w_scale_dev Device pointer to w's dequant scale.
  /// \param stream      Stream to launch on. Does **not** synchronize.
  /// \return            OK, InvalidArgument for shapes this cannot serve,
  ///                    Unimplemented when no FP8 algorithm exists for the
  ///                    shape, or the cuBLAS error.
  Status LinearF8E4M3(const TensorView& x, const TensorView& w,
                      const TensorView& y, const float* x_scale_dev,
                      const float* w_scale_dev, cudaStream_t stream = nullptr);

  /// \brief Builds and caches the plan for one shape without running it.
  ///
  /// \param m, n, k The problem shape, as in `LinearF16`.
  /// \param fp8     Plan the FP8 path rather than the FP16 one. The two are
  ///                cached separately: the layouts differ in element type, so
  ///                an algorithm chosen for one is not valid for the other.
  /// \return        OK, or the error the heuristic reported.
  Status Warm(int64_t m, int64_t n, int64_t k, bool fp8 = false);

  /// \brief Number of distinct shapes currently planned.
  size_t PlanCacheSize() const;

 private:
  struct Impl;

  explicit CublasLtGemm(std::unique_ptr<Impl> impl);

  // Pimpl so that cublasLt.h stays out of this header -- it is included by the
  // bench harness and the tests, neither of which should acquire a cuBLAS
  // dependency to ask for a matrix multiply.
  std::unique_ptr<Impl> impl_;
};

}  // namespace inferx::kernels
