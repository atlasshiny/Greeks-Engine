#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include "models/MonteCarloModel.hpp"
#include "gpu/MonteCarloKernel.cuh"
#include "gpu/error_checking.cuh"

__global__ void setupRNGKernel(curandState_t* states, unsigned long long seed, long numThreads) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numThreads) return;

    // Initialize state per thread
    curand_init(seed, i, 0, &states[i]);
}

__global__ void computeMonteCarloPricingKernel(const Option* option, const MarketParams* mktparams, double* results, int n_simulations,  int n_options, curandState_t* states) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n_options) {
        // Load state to thread register
        curandState_t localState = states[i];

        // Create MonteCarloModel object
        MonteCarloModel model(mktparams[i].S, option[i].K, option[i].T, mktparams[i].r, mktparams[i].sigma, n_simulations);

        // Calculate price using model class
        results[i] = model.price(option[i].type, localState);

        // Save updated RNG state back to global memory
        states[i] = localState;
    }
}

void launchMonteCarloPricingKernel(const Option* h_options, const MarketParams* h_mktparams, double* h_results, int n_simulations, int n_options) {
    Option *d_options;
    MarketParams *d_mktparams;
    double *d_results;
    curandState_t *d_states; // Pointer for RNG states

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_options, n_options * sizeof(Option)));
    CUDA_CHECK(cudaMalloc(&d_mktparams, n_options * sizeof(MarketParams)));
    CUDA_CHECK(cudaMalloc(&d_results, n_options * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_states, n_options * sizeof(curandState_t))); // Allocate states

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_options, h_options, n_options * sizeof(Option), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mktparams, h_mktparams, n_options * sizeof(MarketParams), cudaMemcpyHostToDevice));

    // Grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_options + threadsPerBlock - 1) / threadsPerBlock;

    // Initialize cuRAND States
    setupRNGKernel<<<blocksPerGrid, threadsPerBlock>>>(d_states, 1234ULL, n_options);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // Ensure RNG setup finishes before pricing

    // Launch Monte Carlo Pricing Kernel (Passing d_states!)
    computeMonteCarloPricingKernel<<<blocksPerGrid, threadsPerBlock>>>(
        d_options, d_mktparams, d_results, n_simulations, n_options, d_states
    );
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, n_options * sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_mktparams));
    CUDA_CHECK(cudaFree(d_results));
    CUDA_CHECK(cudaFree(d_states)); // Free states
}