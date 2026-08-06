#pragma once

/// \brief Marks a function callable from both host and device code.
///
/// Expands to `__host__ __device__` when nvcc is compiling the translation
/// unit, and to nothing otherwise. Core headers are included from three kinds
/// of translation unit -- host `.cc` in a CUDA build, host `.cc` in a
/// `--host-only` build, and `.cu` -- and only the third may see the keywords at
/// all.
///
/// The guard is `__CUDACC__`, deliberately, and not `INFERX_WITH_CUDA`. The
/// latter says the *build* has device support and is defined for host `.cc`
/// files too, where `__host__` is not a keyword and the expansion would not
/// compile. `__CUDACC__` says nvcc is the compiler for this translation unit,
/// which is the actual question.
///
/// Annotating a function does not make it device-callable on its own: every
/// function it calls must be reachable from device code too, either annotated
/// or `constexpr` (which `--expt-relaxed-constexpr`, set in
/// cmake/InferXCuda.cmake, makes callable). Adding this to a function whose body
/// calls a host-only one is an nvcc error, not a silent host-side fallback.
#if defined(__CUDACC__)
#define INFERX_HOST_DEVICE __host__ __device__
#else
#define INFERX_HOST_DEVICE
#endif
