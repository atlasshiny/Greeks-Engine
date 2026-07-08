#pragma once
#include "Greeks.hpp"

// GPU macro for CUDA compatibility
#ifdef __CUDACC__
    #define HOST_DEVICE __host__ __device__
#else
    #define HOST_DEVICE
#endif

class IPricingModel {
public:
    HOST_DEVICE virtual ~IPricingModel() = default;

    // Method to calculate the price of an option
    HOST_DEVICE virtual double price(int optionType) const = 0;

    // Method to calculate the Greeks of an option
    HOST_DEVICE virtual Greeks calculateGreeks(int optionType) const = 0;
};