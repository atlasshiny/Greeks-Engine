#pragma once

// GPU macro for CUDA compatibility
#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

