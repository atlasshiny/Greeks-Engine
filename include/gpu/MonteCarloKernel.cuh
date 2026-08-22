#pragma once
#include "Option.hpp"
#include "MarketParameters.hpp"

// The kernel that executes the pricing code on the GPU
__global__ void computeMonteCarloPricingKernel(const Option* option, const MarketParams* mktparams, double* results, int n_simulations, int n_options, curandState_t* states);

// The Bridge Function for the pricing calculation: Orchestrates memory and execution
void launchMonteCarloPricingKernel(const Option* h_options, const MarketParams* h_mktparams, double* h_results, int n_simulations, int n_options);