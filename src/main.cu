#include <iostream>
#include <vector>
#include "gpu/BSMKernel.cuh"
#include "models/BSMModel.hpp"
#include "models/BinomialTreeModel.hpp"
#include "Option.hpp"
#include "Greeks.hpp"
#include "MarketParameters.hpp"
#include <cuda_runtime.h>

// Forward declarations
void launchBSMGreeksKernel(const Option* h_options, const MarketParams* h_mktparams, Greeks* h_results, int n);

void BSMModelImplementation(){

}

int main() {
    int n = 10000000; // Number of options to process
    std::vector<Option> options(n, {100.0, 1.0, 0});
    std::vector<MarketParams> mktparams(n, {100.0, 0.05, 0.2, 0.0});
    std::vector<Greeks> results(n);

    // Launch parallel computation on GPU
    launchBSMGreeksKernel(options.data(), mktparams.data(), results.data(), n);

    std::cout << "Successfully processed " << n << " options on the GPU." << std::endl;
    std::cout << "Delta of first option: " << results[0].delta << std::endl;

    return 0;
}