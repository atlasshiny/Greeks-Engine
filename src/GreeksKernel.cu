#include <cuda_runtime.h>
#include "models/BSMModel.hpp"
#include "Option.hpp"
#include "Greeks.hpp"
#include "gpu/GreeksKernel.cuh"
#include "gpu/error_checking.cuh"

// The kernel that executes the code on the GPU
__global__ void computeGreeksKernel(const Option* options, Greeks* results, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Instantiate the header-only BSMModel on the GPU thread
        BSMModel model(options[i].S, options[i].K, options[i].T, options[i].r, options[i].sigma);
        
        // Calculate Greeks using the header-file code
        results[i] = model.calculateGreeks(options[i].type);

    }
}

// The Bridge Function: Orchestrates memory and execution
void launchGreeksKernel(const Option* h_options, Greeks* h_results, int n) {
    Option *d_options;
    Greeks *d_results;

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_options, n * sizeof(Option)));
    CUDA_CHECK(cudaMalloc(&d_results, n * sizeof(Greeks)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_options, h_options, n * sizeof(Option), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel on GPU
    computeGreeksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_results, n);

    // Check for errors after launching the kernel
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, n * sizeof(Greeks), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_results));
}
