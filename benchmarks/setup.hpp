#pragma once
#include <vector>
#include "Option.hpp"
#include "MarketParameters.hpp"
#include "Greeks.hpp"

struct BenchmarkBatch {
    std::vector<Option> options;
    std::vector<MarketParams> mktparams;
    std::vector<Greeks> results;
};

BenchmarkBatch generateBenchmarkBatch(size_t n) {
    BenchmarkBatch data;
    data.options = std::vector<Option>(n, {100.0, 1.0, 0}); // Example: Spot=100, Strike=100, T=1 year, r=5%, sigma=20%, type=Call
    data.mktparams = std::vector<MarketParams>(n, {100.0, 0.05, 0.2, 0.0});
    data.results = std::vector<Greeks>(n); // Preallocate results

    return data;
}