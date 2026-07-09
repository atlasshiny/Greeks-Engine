#include "gpu/BinomialTreeKernel.cuh"
#include "gpu/error_checking.cuh"
#include "models/BinomialTreeModel.hpp"
#include <cuda_runtime.h>

// The kernel that executes the Greeks code on the GPU
__global__ void computeBinomialGreeksKernel(const Option* options, const MarketParams* mktparams, Greeks* results, int n_steps, int n_options, double* buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_options) {
        // Instantiate the header-only BinomialTreeModel on the GPU thread
        BinomialTreeModel model(mktparams[i].S, options[i].K, options[i].T, mktparams[i].r, mktparams[i].sigma, mktparams[i].q, n_steps, options[i].isAmerican);
        
        // Slice the buffer per thread so that it is n_steps + 1 size.
        // Because 'i' is unique to each thread, it ensures that the buffer slice isn't slicing something that is already passed into another thread
        double* buffer_slice = &buffer[i * (n_steps + 1)];

        // Calculate Greeks using the header-file code (when it is implemented)
        results[i] = model.calculateGreeks(options[i].type, buffer_slice, 0.001);

    }
};

// The Bridge Function: Orchestrates memory and execution
void launchBinomialGreeksKernel(const Option* h_options, const MarketParams* h_mktparams, Greeks* h_results, int n_steps, int n_options) {
    Option *d_options;
    MarketParams *d_mktparams;
    double *d_buffer;
    Greeks *d_results;

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_options, n_options * sizeof(Option)));
    CUDA_CHECK(cudaMalloc(&d_mktparams, n_options * sizeof(MarketParams)));
    CUDA_CHECK(cudaMalloc(&d_buffer, n_options * (n_steps + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_results, n_options * sizeof(Greeks)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_options, h_options, n_options * sizeof(Option), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mktparams, h_mktparams, n_options * sizeof(MarketParams), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_options + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel on GPU
    computeBinomialGreeksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_mktparams, d_results, n_steps, n_options, d_buffer);

    // Check for errors after launching the kernel
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, n_options * sizeof(Greeks), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_mktparams));
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_results));
}

// The kernel that executes the pricing code on the GPU
__global__ void computeBinomialPricingKernel(const Option* options, const MarketParams* mktparams, double* results, int n_steps, int n_options, double* buffer) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_options) {
        // Instantiate the header-only BinomialTreeModel on the GPU thread
        BinomialTreeModel model(mktparams[i].S, options[i].K, options[i].T, mktparams[i].r, mktparams[i].sigma, mktparams[i].q, n_steps, options[i].isAmerican);
        
        // Slice the buffer per thread so that it is n_steps + 1 size.
        // Because 'i' is unique to each thread, it ensures that the buffer slice isn't slicing something that is already passed into another thread
        double* buffer_slice = &buffer[i * (n_steps + 1)];

        // Calculate price using the header-file code
        results[i] = model.price(options[i].type, buffer_slice);

    }
}

// The Bridge Function: Orchestrates memory and execution
void launchBinomialPricingKernel(const Option* h_options, const MarketParams* h_mktparams, double* h_results, int n_steps, int n_options) {
    Option *d_options;
    MarketParams *d_mktparams;
    double *d_buffer;
    double *d_results;

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_options, n_options * sizeof(Option)));
    CUDA_CHECK(cudaMalloc(&d_mktparams, n_options * sizeof(MarketParams)));
    CUDA_CHECK(cudaMalloc(&d_buffer, n_options * (n_steps + 1) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_results, n_options * sizeof(double)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_options, h_options, n_options * sizeof(Option), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mktparams, h_mktparams, n_options * sizeof(MarketParams), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n_options + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel on GPU
    computeBinomialPricingKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_mktparams, d_results, n_steps, n_options, d_buffer);

    // Check for errors after launching the kernel
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, n_options * sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_mktparams));
    CUDA_CHECK(cudaFree(d_buffer));
    CUDA_CHECK(cudaFree(d_results));
}