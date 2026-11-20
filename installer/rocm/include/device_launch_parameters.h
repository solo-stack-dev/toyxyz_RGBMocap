// CUDA declares threadIdx/blockIdx here; HIP declares them in hip_runtime.h,
// which hipified sources already include. Empty on purpose (AMD ROCm builds).
#pragma once
#include <hip/hip_runtime.h>
