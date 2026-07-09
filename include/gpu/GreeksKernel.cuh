#pragma once
#include "Option.hpp"
#include "Greeks.hpp"

// The kernel that executes the Greeks calculation code on the GPU
__global__ void computeGreeksKernel(const Option* options, Greeks* results, int n);

// The kernel that executes the pricing code on the GPU
__global__ void computePriceKernel(const Option* option, double* results, int n);

// The Bridge Function for the Greeks calculation: Orchestrates memory and execution
void launchGreeksKernel(const Option* h_options, Greeks* h_results, int n);

// The Bridge Function for the pricing calculation: Orchestrates memory and execution
void launchPricingKernel(const Option* h_options, double* h_results, int n);