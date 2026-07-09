#pragma once

// Macro for CUDA compatibility
#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

// Macro for switching from std math functions to CUDA math functions when compiling with NVCC
#ifdef __CUDACC__
    #define MATH_FUNC(func) func##f
#else
    #define MATH_FUNC(func) std::func
#endif