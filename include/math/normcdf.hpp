#pragma once
#include <cmath>

// GPU macro for CUDA compatibility
#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

namespace MathUtils {
    HOST_DEVICE inline double normcdf(double x) {
        return 0.5 * (1.0 + std::erf(x / std::sqrt(2.0)));
    }
}