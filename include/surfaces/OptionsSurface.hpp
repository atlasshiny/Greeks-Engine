#pragma once

#include <vector>
#include <fstream>
#include "macros.hpp"

// CUDA Kernel Definition (Parsed only during CUDA compilation)
#ifdef __CUDACC__
#include <cuda_runtime.h>

template <typename PricingPolicy>
__global__ void generateOptionSurfaceKernel(
    float* __restrict__ d_surface,
    int numS, int numVol,
    float S_min, float S_step,
    float vol_min, float vol_step,
    float T, float r, float K, int optionType,
    PricingPolicy policy) 
{
    int s_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int v_idx = blockIdx.y * blockDim.y + threadIdx.y;

    if (s_idx < numS && v_idx < numVol) {
        float S = S_min + s_idx * S_step;
        float vol = vol_min + v_idx * vol_step;

        // Shared policy call
        d_surface[v_idx * numS + s_idx] = policy(S, K, T, r, vol, optionType);
    }
}
#endif

// Cross-Platform Surface Engine Interface
class OptionSurfaceEngine {
public:
    // CPU Generator
    template <typename PricingPolicy>
    static std::vector<float> generateCPU(
        float* surface,
        int numS, int numVol,
        float S_min, float S_step,
        float vol_min, float vol_step,
        float T, float r, float K, int optionType,
        PricingPolicy policy) 
    {
        for (int v_idx = 0; v_idx < numVol; ++v_idx) {
            for (int s_idx = 0; s_idx < numS; ++s_idx) {
                float S = S_min + s_idx * S_step;
                float vol = vol_min + v_idx * vol_step;

                // Exact same shared policy call
                surface[v_idx * numS + s_idx] = policy(S, K, T, r, vol, optionType);
            }
        }

        // Save the generated surface to a file for verification
        saveSurfaceToFile(surface, numS, numVol, "/Greeks-Engine/option_surface_cpu.bin");
        
        return std::vector<float>(surface, surface + numS * numVol);
    }

#ifdef __CUDACC__
    // GPU Acceleration Launcher
    // This version is copies results back to host and frees GPU memory
    template <typename PricingPolicy>
    static std::vector<float> generateGPU(
        int numS, int numVol,
        float S_min, float S_step,
        float vol_min, float vol_step,
        float T, float r, float K, int optionType,
        PricingPolicy policy,
        dim3 blockSize = dim3(16, 16)) 
    {
        // Allocate host vector to return
        size_t totalElements = numS * numVol;
        size_t bytes = totalElements * sizeof(float);
        std::vector<float> h_surface(totalElements);

        // Allocate temporary GPU memory
        float* d_surface = nullptr;
        cudaMalloc(&d_surface, bytes);

        dim3 gridSize((numS + blockSize.x - 1) / blockSize.x, 
                     (numVol + blockSize.y - 1) / blockSize.y);

        generateOptionSurfaceKernel<<<gridSize, blockSize>>>(
            d_surface, numS, numVol, S_min, S_step, vol_min, vol_step, 
            T, r, K, optionType, policy
        );
        cudaDeviceSynchronize();

        // Copy back to host vector
        cudaMemcpy(h_surface.data(), d_surface, bytes, cudaMemcpyDeviceToHost);

        // Cleanup VRAM
        cudaFree(d_surface);

        saveSurfaceToFile(h_surface.data(), numS, numVol, "/Greeks-Engine/option_surface_gpu.bin");

        return h_surface;
    }

    // GPU Acceleration Launcher
    // This version keeps data on the GPU for future kernel operations
    template <typename PricingPolicy>
    static void generateGPUInPlace(
        float* d_surface,
        int numS, int numVol,
        float S_min, float S_step,
        float vol_min, float vol_step,
        float T, float r, float K, int optionType,
        PricingPolicy policy,
        dim3 blockSize = dim3(16, 16)) 
    {
        dim3 gridSize((numS + blockSize.x - 1) / blockSize.x, 
                    (numVol + blockSize.y - 1) / blockSize.y);

        generateOptionSurfaceKernel<<<gridSize, blockSize>>>(
            d_surface, numS, numVol, S_min, S_step, vol_min, vol_step, 
            T, r, K, optionType, policy
        );
        cudaDeviceSynchronize();
    }

#endif

private:
    OptionSurfaceEngine() = delete; // Prevent instantiation

    // Helper method for saving the generated surface to a file (used in CPU & GPU generation)

    static void saveSurfaceToFile(const float* surface, int numS, int numVol, const std::string& filename) {
        std::ofstream outFile(filename, std::ios::binary);
        if (outFile.is_open()) {
            outFile.write(reinterpret_cast<const char*>(surface), numS * numVol * sizeof(float));
            outFile.close();
        } else {
            std::cerr << "Unable to open file for writing option surface." << std::endl;
        }
    };
};