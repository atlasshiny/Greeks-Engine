#pragma once
#include <cuda_runtime.h>
#include "Option.hpp"
#include "MarketParameters.hpp"
#include "Greeks.hpp"

// The kernel that executes the Greeks calculation code on the GPU
__global__ void computeBSMGreeksKernel(const Option* options, const MarketParams* mktparams, Greeks* results, int n);

// The Bridge Function for the Greeks calculation: Orchestrates memory and execution
void launchBSMGreeksKernel(const Option* h_options, const MarketParams* h_mktparams, Greeks* h_results, int n);

// The kernel that executes the pricing code on the GPU
__global__ void computeBSMPricingKernel(const Option* option, const MarketParams* mktparams, double* results, int n);

// The Bridge Function for the pricing calculation: Orchestrates memory and execution
void launchBSMPricingKernel(const Option* h_options, const MarketParams* h_mktparams, double* h_results, int n);

// Torch Tensor Operations
__global__ void computeBSMPricingTensorKernel(
    const float* __restrict__ S, 
    const float* __restrict__ K, 
    const float* __restrict__ T, 
    const float* __restrict__ r, 
    const float* __restrict__ sigma, 
    const int* __restrict__ option_type, 
    float* __restrict__ results, 
    int n
);

void launchBSMPricingTensorKernel(
    const float* d_S, 
    const float* d_K, 
    const float* d_T,
    const float* d_r, 
    const float* d_sigma, 
    const int* d_option_type,
    float* d_results, 
    int n, 
    cudaStream_t stream = 0
);