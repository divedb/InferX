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

namespace inferx::kernels {

/// \brief Multiplies `n` floats in device memory by `scale`, in place.
///
/// Deliberately takes raw pointers rather than a TensorView. The view cannot
/// carry this call today -- its accessors are host-only, so a kernel cannot ask
/// it for its data pointer or extent (see the note in smoke.cu) -- and the
/// point here is to isolate the toolchain, not to depend on that being
/// resolved.
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

}  // namespace inferx::kernels

#endif  // INFERX_KERNELS_SMOKE_H_
