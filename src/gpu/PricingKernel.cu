#include "gpu/PricingKernel.cuh"
#include "gpu/error_checking.cuh"
#include "models/BSMModel.hpp"
#include <cuda_runtime.h>

// The kernel that executes the pricing code on the GPU
__global__ void computePriceKernel(const Option* options, double* results, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Instantiate the header-only BSMModel on the GPU thread
        BSMModel model(options[i].S, options[i].K, options[i].T, options[i].r, options[i].sigma);
        
        // Calculate price using the header-file code
        results[i] = model.price(options[i].type);

    }
}

// The Bridge Function: Orchestrates memory and execution
void launchPricingKernel(const Option* h_options, double* h_results, int n) {
    Option *d_options;
    double *d_results;

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_options, n * sizeof(Option)));
    CUDA_CHECK(cudaMalloc(&d_results, n * sizeof(double)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_options, h_options, n * sizeof(Option), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel on GPU
    computePriceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_results, n);

    // Check for errors after launching the kernel
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_results));
}