// Device-path smoke test.
//
// This is not engine functionality. It exists so that "does nvcc compile, link,
// and launch inside this build" is answered by a test rather than discovered
// halfway through the first CUTLASS integration. The whole device toolchain --
// nvcc driving the host compiler, the sm_89 codegen, the CUDA flags on
// inferx::flags, linking a .cu object into a gtest binary, and our own core
// headers surviving nvcc's frontend -- is exercised by the one launch below.
//
// It is the M1 canary: when CUTLASS lands and something in the device build is
// wrong, this test failing (or not) says whether the problem is CUTLASS or the
// toolchain underneath it. Delete it only when there are real kernels covering
// the same ground.

#ifndef INFERX_KERNELS_SMOKE_H_
#define INFERX_KERNELS_SMOKE_H_

#include <cstdint>

#include "inferx/core/status.h"
#include "inferx/core/tensor_view.h"

namespace inferx::ops {

/// \brief Multiplies `n` floats in device memory by `scale`, in place.
///
/// Takes raw pointers so that this overload isolates the toolchain and nothing
/// else: if it passes and the TensorView overload below does not, the problem
/// is the view, not nvcc.
///
/// Synchronizes before returning, so a launch failure is reported as an error
/// rather than surfacing at some later unrelated CUDA call. That is the correct
/// trade for a diagnostic and the wrong one for anything on the step path.
///
/// \param device_data Device pointer to `n` floats. May be null only if n == 0.
/// \param n           Number of elements to scale.
/// \param scale       The factor to multiply by.
/// \return            OK, or the CUDA error that the launch or sync reported.
Status LaunchScale(float* device_data, int64_t n, float scale);

/// \brief The same operation, driven entirely by a TensorView.
///
/// The view is passed to the kernel by value and read there through its
/// INFERX_HOST_DEVICE accessors. That pairing -- trivially copyable to get in,
/// annotated accessors to be useful once in -- is the §7.1 kernel-boundary
/// claim, and this is the launch that tests it rather than asserting it.
///
/// \param view  An f32 view on a CUDA device. Empty views are a no-op.
/// \param scale The factor to multiply by.
/// \return      OK, InvalidArgument for a view this cannot act on, or the CUDA
///              error that the launch or sync reported.
Status LaunchScale(const TensorView& view, float scale);

}  // namespace inferx::ops

#endif  // INFERX_KERNELS_SMOKE_H_
