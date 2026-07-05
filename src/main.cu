#include <iostream>
#include <vector>
#include "models/BSMModel.hpp"
#include "Option.hpp"
#include "Greeks.hpp"
#include <cuda_runtime.h>

// Forward declaration
__global__ void computeGreeksKernel(const Option* options, Greeks* results, int n);

// The Bridge Function: Orchestrates memory and execution
void launchGreeksKernel(const Option* h_options, Greeks* h_results, int n) {
    Option *d_options;
    Greeks *d_results;

    // Allocate memory on GPU
    cudaMalloc(&d_options, n * sizeof(Option));
    cudaMalloc(&d_results, n * sizeof(Greeks));

    // Copy data to GPU
    cudaMemcpy(d_options, h_options, n * sizeof(Option), cudaMemcpyHostToDevice);

    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel on GPU
    computeGreeksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_results, n);

    // Copy results back to host
    cudaMemcpy(h_results, d_results, n * sizeof(Greeks), cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_options);
    cudaFree(d_results);
}

int main() {
    int n = 1000; // Number of options to process
    std::vector<Option> options(n, {100.0, 100.0, 1.0, 0.05, 0.2, 0});
    std::vector<Greeks> results(n);

    // Launch parallel computation on GPU
    launchGreeksKernel(options.data(), results.data(), n);

    std::cout << "Successfully processed " << n << " options on the GPU." << std::endl;
    std::cout << "Delta of first option: " << results[0].delta << std::endl;

    return 0;
}