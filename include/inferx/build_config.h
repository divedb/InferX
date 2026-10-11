#pragma once

namespace inferx {

/// \brief Compile-time backend availability for this build.
///
/// `kHasX` is true when the X runtime was linked in, i.e. the build can
/// create and launch work on devices of that kind. It says nothing about
/// whether hardware is present; that is checked when a device is activated.
///
/// Set by CMake via `INFERX_WITH_<BACKEND>` definitions, which must be applied
/// at the target level so every translation unit sees the same values.
/// Code should branch on these constants (`if constexpr (kHasCuda)`) instead of
/// writing `#ifdef INFERX_WITH_CUDA` directly.
#ifdef INFERX_WITH_CUDA
inline constexpr bool kHasCuda = true;
#else
inline constexpr bool kHasCuda = false;
#endif

#ifdef INFERX_WITH_HIP
inline constexpr bool kHasHip = true;
#else
inline constexpr bool kHasHip = false;
#endif

}  // namespace inferx