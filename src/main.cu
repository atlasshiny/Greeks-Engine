#include <iostream>
#include <vector>
#include "gpu/GreeksKernel.cuh"
#include "models/BSMModel.hpp"
#include "Option.hpp"
#include "Greeks.hpp"
#include <cuda_runtime.h>

// Forward declarations
__global__ void computeGreeksKernel(const Option* options, Greeks* results, int n);
void launchGreeksKernel(const Option* h_options, Greeks* h_results, int n);

int main() {
    int n = 10000000; // Number of options to process
    std::vector<Option> options(n, {100.0, 100.0, 1.0, 0.05, 0.2, 0});
    std::vector<Greeks> results(n);

    // Launch parallel computation on GPU
    launchGreeksKernel(options.data(), results.data(), n);

    std::cout << "Successfully processed " << n << " options on the GPU." << std::endl;
    std::cout << "Delta of first option: " << results[0].delta << std::endl;

    return 0;
}