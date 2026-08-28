#pragma once

// Macro for CUDA compatibility
#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
    #define DEVICE __device__
    #define HOST __host__
#else
    #define HOST_DEVICE
    #define DEVICE
    #define HOST
#endif

// Macro for model policy structs
#ifdef __CUDACC__
    #define MODEL_POLICY __host__ __device__ __forceinline__
#else
    #define MODEL_POLICY inline
#endif

// Macro for switching from std math functions to CUDA math functions when compiling with NVCC
#ifdef __CUDACC__
    #define MATH_FUNC(func) func##f
#else
    #define MATH_FUNC(func) std::func
#endif