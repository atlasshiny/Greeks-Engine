#pragma once
#include "Option.hpp"
#include "MarketParameters.hpp"
#include "Greeks.hpp"

// The kernel that executes the Greeks calculation code on the GPU
__global__ void computeBinomialGreeksKernel(const Option* options, const MarketParams* mktparams, Greeks* results, int n_steps, int n_options, double* buffer);

// The Bridge Function for the Greeks calculation: Orchestrates memory and execution
void launchBinomialGreeksKernel(const Option* h_options, const MarketParams* h_mktparams, Greeks* h_results, int n_steps, int n_options);

// The kernel that executes the pricing code on the GPU
__global__ void computeBinomialPricingKernel(const Option* option, const MarketParams* mktparams, double* results, int n_steps, int n_options, double* buffer);

// The Bridge Function for the pricing calculation: Orchestrates memory and execution
void launchBinomialPricingKernel(const Option* h_options, const MarketParams* h_mktparams, double* h_results, int n_steps, int n_options);
