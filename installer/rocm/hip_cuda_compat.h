// Force-include shim for compiling unmodified CUDA sources with HIP (ROCm 7.x).
// Use: CUDAExtension(..., extra_compile_args={"nvcc": ["-include", "installer/rocm/hip_cuda_compat.h", ...]})
//
// HIP's __*_sync warp intrinsics static_assert that the mask is a 64-bit integer,
// so the ubiquitous CUDA idiom  __shfl_down_sync(0xffffffff, v, off)  fails with
// "The mask must be a 64-bit integer". These more-specialised overloads accept a
// 32-bit mask (unsigned or int) and widen it. Correct on wave32 GPUs (RDNA, e.g.
// gfx1100) where HIP ignores the upper 32 bits anyway. On wave64 (CDNA/GCN) a
// 32-bit full mask would only name lanes 0-31: there, widen 0xffffffff to ~0ull.
#pragma once
#if defined(__HIPCC__) && defined(__HIP_PLATFORM_AMD__) && !defined(HIP_DISABLE_WARP_SYNC_BUILTINS)
#include <hip/hip_runtime.h>

namespace rx_hip_compat {
__device__ inline unsigned long long widen(unsigned int m) {
  // full 32-bit mask on a wave64 device -> full 64-bit mask
  return (m == 0xffffffffu && warpSize == 64) ? ~0ull : static_cast<unsigned long long>(m);
}
}  // namespace rx_hip_compat

#define RX_HIP_MASK_SHIMS(MT)                                                                        \
  template <typename T>                                                                              \
  __device__ inline T __shfl_sync(MT m, T v, int src, int w = warpSize) {                            \
    return __shfl_sync(rx_hip_compat::widen((unsigned)m), v, src, w);                                \
  }                                                                                                  \
  template <typename T>                                                                              \
  __device__ inline T __shfl_up_sync(MT m, T v, unsigned int d, int w = warpSize) {                  \
    return __shfl_up_sync(rx_hip_compat::widen((unsigned)m), v, d, w);                               \
  }                                                                                                  \
  template <typename T>                                                                              \
  __device__ inline T __shfl_down_sync(MT m, T v, unsigned int d, int w = warpSize) {                \
    return __shfl_down_sync(rx_hip_compat::widen((unsigned)m), v, d, w);                             \
  }                                                                                                  \
  template <typename T>                                                                              \
  __device__ inline T __shfl_xor_sync(MT m, T v, int lm, int w = warpSize) {                         \
    return __shfl_xor_sync(rx_hip_compat::widen((unsigned)m), v, lm, w);                             \
  }                                                                                                  \
  __device__ inline unsigned long long __ballot_sync(MT m, int p) {                                  \
    return __ballot_sync(rx_hip_compat::widen((unsigned)m), p);                                      \
  }                                                                                                  \
  __device__ inline int __any_sync(MT m, int p) { return __any_sync(rx_hip_compat::widen((unsigned)m), p); } \
  __device__ inline int __all_sync(MT m, int p) { return __all_sync(rx_hip_compat::widen((unsigned)m), p); } \
  __device__ inline void __syncwarp(MT m) { __syncwarp(rx_hip_compat::widen((unsigned)m)); }        \
  template <typename T>                                                                              \
  __device__ inline unsigned long long __match_any_sync(MT m, T v) {                                 \
    return __match_any_sync(rx_hip_compat::widen((unsigned)m), v);                                   \
  }                                                                                                  \
  template <typename T>                                                                              \
  __device__ inline unsigned long long __match_all_sync(MT m, T v, int* p) {                         \
    return __match_all_sync(rx_hip_compat::widen((unsigned)m), v, p);                                \
  }

RX_HIP_MASK_SHIMS(unsigned int)
RX_HIP_MASK_SHIMS(int)
#undef RX_HIP_MASK_SHIMS

#endif
