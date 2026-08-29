#include <cuda_runtime.h>
#include "models/BSMModel.hpp"
#include "gpu/BSMKernel.cuh"
#include "gpu/error_checking.cuh"

// The kernel that executes the Greeks code on the GPU
__global__ void computeBSMGreeksKernel(const Option* options, const MarketParams* mktparams, Greeks* results, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Instantiate the header-only BSMModel on the GPU thread
        BSMModel model(mktparams[i].S, options[i].K, options[i].T, mktparams[i].r, mktparams[i].sigma);
        
        // Calculate Greeks using the header-file code
        results[i] = model.calculateGreeks(options[i].type);

    }
}

// The Bridge Function: Orchestrates memory and execution
// Ensure to use pinned memory for the host memory (the parameters being passed in) by using cudaHostAlloc()
void launchBSMGreeksKernel(const Option* h_options, const MarketParams* h_mktparams, Greeks* h_results, int n) {
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
    computeBSMGreeksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_mktparams, d_results, n);

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
__global__ void computeBSMPricingKernel(const Option* options, const MarketParams* mktparams, double* results, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Instantiate the header-only BSMModel on the GPU thread
        BSMModel model(mktparams[i].S, options[i].K, options[i].T, mktparams[i].r, mktparams[i].sigma);
        
        // Calculate price using the header-file code
        results[i] = model.price(options[i].type);

    }
}

// The Bridge Function: Orchestrates memory and execution
// Ensure to use pinned memory for the host memory (the parameters being passed in) by using cudaHostAlloc()
void launchBSMPricingKernel(const Option* h_options, const MarketParams* h_mktparams, double* h_results, int n) {
    Option *d_options;
    MarketParams *d_mktparams;
    double *d_results;

    // Allocate memory on GPU
    CUDA_CHECK(cudaMalloc(&d_options, n * sizeof(Option)));
    CUDA_CHECK(cudaMalloc(&d_mktparams, n * sizeof(MarketParams)));
    CUDA_CHECK(cudaMalloc(&d_results, n * sizeof(double)));

    // Copy data to GPU
    CUDA_CHECK(cudaMemcpy(d_options, h_options, n * sizeof(Option), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mktparams, h_mktparams, n * sizeof(MarketParams), cudaMemcpyHostToDevice));

    // Calculate grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel on GPU
    computeBSMPricingKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_mktparams, d_results, n);

    // Check for errors after launching the kernel
    CUDA_CHECK(cudaGetLastError());

    // Copy results back to host
    CUDA_CHECK(cudaMemcpy(h_results, d_results, n * sizeof(double), cudaMemcpyDeviceToHost));

    // Free GPU memory
    CUDA_CHECK(cudaFree(d_options));
    CUDA_CHECK(cudaFree(d_mktparams));
    CUDA_CHECK(cudaFree(d_results));
}

// Torch Kernels
__global__ void computeBSMPricingTensorKernel(
    const double* __restrict__ S,
    const double* __restrict__ K,
    const double* __restrict__ T,
    const double* __restrict__ r,
    const double* __restrict__ sigma,
    const int* __restrict__ option_type,
    double* __restrict__ results,
    int n) 
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        double d1_val = BSMMath::d1(S[i], K[i], T[i], r[i], sigma[i]);
        double d2_val = BSMMath::d2(d1_val, sigma[i], T[i]);
        results[i] = (option_type[i] == 0) 
            ? BSMMath::callPrice(S[i], K[i], T[i], r[i], d1_val, d2_val)
            : BSMMath::putPrice(S[i], K[i], T[i], r[i], d1_val, d2_val);
    }
}

// Direct launcher: Operates directly on existing GPU VRAM allocations
void launchBSMPricingTensorKernel(
    const double* d_S, 
    const double* d_K, 
    const double* d_T, 
    const double* d_r, 
    const double* d_sigma, 
    const int* d_option_type,
    double* d_results, 
    int n,
    cudaStream_t stream) 
{
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    computeBSMPricingTensorKernel<<<blocksPerGrid, threadsPerBlock, 0, stream>>>(
        d_S, d_K, d_T, d_r, d_sigma, d_option_type, d_results, n);

    CUDA_CHECK(cudaGetLastError());
}