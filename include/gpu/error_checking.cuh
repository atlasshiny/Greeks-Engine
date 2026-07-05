#pragma once
#include <cuda_runtime.h>
#include <iostream>
#include <string>

#define CUDA_CHECK(val) check((val), #val, __FILE__, __LINE__)

void check(cudaError_t err, const char* func, const char* file, int line) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line << std::endl;
        std::cerr << func << " failed with error " << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}