#pragma once
#include <vector>
#include "Option.hpp"
#include "MarketParameters.hpp"
#include "Greeks.hpp"

struct BenchmarkBatch {
    std::vector<Option> options;
    std::vector<MarketParams> mktparams;
    std::vector<Greeks> greek_results;
    std::vector<double> price_results;
};

BenchmarkBatch generateBenchmarkBatch(size_t n) {
    BenchmarkBatch data;
    data.options = std::vector<Option>(n, {100.0, 1.0, 0, false}); // Example: Strike=100, T=1 year, type=Call, isAmerican=false
    data.mktparams = std::vector<MarketParams>(n, {100.0, 0.05, 0.2, 0.0}); // Example: Spot=100, r=5%, sigma=20%, q=0%
    data.greek_results = std::vector<Greeks>(n); // Preallocate Greek results
    data.price_results = std::vector<double>(n); // Preallocate price results

    return data;
}