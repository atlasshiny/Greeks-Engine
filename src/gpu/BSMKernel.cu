#include <cuda_runtime.h>
#include "models/BSMModel.hpp"
#include "gpu/BSMKernel.cuh"
#include "gpu/error_checking.cuh"

// The kernel that executes the Greeks code on the GPU
__global__ void computeGreeksKernel(const Option* options, const MarketParams* mktparams, Greeks* results, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Instantiate the header-only BSMModel on the GPU thread
        BSMModel model(mktparams[i].S, options[i].K, options[i].T, mktparams[i].r, mktparams[i].sigma);
        
        // Calculate Greeks using the header-file code
        results[i] = model.calculateGreeks(options[i].type);

    }
}

// The Bridge Function: Orchestrates memory and execution
void launchGreeksKernel(const Option* h_options, const MarketParams* h_mktparams, Greeks* h_results, int n) {
    Option *d_options;
    MarketParams *d_mktparams;
    Greeks *d_results;

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_options, n * sizeof(Option)));
    CUDA_CHECK(cudaMalloc(&d_mktparams, n * sizeof(MarketParams)));
    CUDA_CHECK(cudaMalloc(&d_results, n * sizeof(Greeks)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_options, h_options, n * sizeof(Option), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mktparams, h_mktparams, n * sizeof(MarketParams), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel on GPU
    computeGreeksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_mktparams, d_results, n);

    // Check for errors after launching the kernel
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, n * sizeof(Greeks), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_mktparams));
    CUDA_CHECK(cudaFree(d_results));
}

// The kernel that executes the pricing code on the GPU
__global__ void computePricingKernel(const Option* options, const MarketParams* mktparams, double* results, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Instantiate the header-only BSMModel on the GPU thread
        BSMModel model(mktparams[i].S, options[i].K, options[i].T, mktparams[i].r, mktparams[i].sigma);
        
        // Calculate price using the header-file code
        results[i] = model.price(options[i].type);

    }
}

// The Bridge Function: Orchestrates memory and execution
void launchPricingKernel(const Option* h_options, const MarketParams* h_mktparams, double* h_results, int n) {
    Option *d_options;
    MarketParams *d_mktparams;
    double *d_results;

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_options, n * sizeof(Option)));
    CUDA_CHECK(cudaMalloc(&d_mktparams, n * sizeof(MarketParams)));
    CUDA_CHECK(cudaMalloc(&d_results, n * sizeof(double)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_options, h_options, n * sizeof(Option), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel on GPU
    computePricingKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_mktparams, d_results, n);

    // Check for errors after launching the kernel
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_results));
}