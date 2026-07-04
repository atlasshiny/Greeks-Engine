#pragma once
#include <cmath>
#include <numbers>

// GPU macro for CUDA compatibility
#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

// Define PI as a constant using std::numbers::pi
inline constexpr double PI = std::numbers::pi;

namespace MathUtils {
    HOST_DEVICE inline double normpdf(double x) {
        return std::exp(-0.5 * x * x) / std::sqrt(2 * PI);
    }
}