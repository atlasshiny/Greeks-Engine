#include <cuda_runtime.h>
#include "models/BSMModel.hpp"
#include "Option.hpp"
#include "Greeks.hpp"

// The kernel that executes the code on the GPU
__global__ void computeGreeksKernel(const Option* options, Greeks* results, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        // Instantiate the header-only BSMModel on the GPU thread
        BSMModel model(options[i].S, options[i].K, options[i].T, options[i].r, options[i].sigma);
        
        // Calculate Greeks using the header-file code
        if (options[i].type == 0) {
            results[i] = model.callGreeks();
        } else {
            results[i] = model.putGreeks();
        }
    }
}
