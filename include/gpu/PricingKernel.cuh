#pragma once
#include "Option.hpp"
#include "Greeks.hpp"

// The kernel that executes the pricing code on the GPU
__global__ void computePriceKernel(const Option* option, double* results, int n);

// The Bridge Function for the pricing calculation: Orchestrates memory and execution
void launchPricingKernel(const Option* h_options, double* h_results, int n);
