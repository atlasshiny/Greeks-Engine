#include "models/BSMModel.hpp"
#include "setup.hpp"
#include <benchmark/benchmark.h>

static void BM_CPU_BSM(benchmark::State& state) {

    int n = static_cast<int>(state.range()); // Get the number of test cases from the benchmark state

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    BenchmarkBatch data = generateBenchmarkBatch(warmup_size);

    for (size_t i = 0; i < data.options.size(); ++i) {
        BSMModel model(data.options[i].S, data.options[i].K, data.options[i].T, data.options[i].r, data.options[i].sigma);
        if (data.options[i].type == 0) {
            data.results[i] = model.callGreeks();
        } else {
            data.results[i] = model.putGreeks();
        }
    }

    // Generate test inputs
    BenchmarkBatch data = generateBenchmarkBatch(n); // Generate n test cases

    for (auto _ : state) {
        // Run the BSM model on each option and store the results
        for (size_t i = 0; i < data.options.size(); ++i) {
            BSMModel model(data.options[i].S, data.options[i].K, data.options[i].T, data.options[i].r, data.options[i].sigma);

            if (data.options[i].type == 0) {
                data.results[i] = model.callGreeks();
            } else {
                data.results[i] = model.putGreeks();
            }
        }
    }
}

BENCHMARK(BM_CPU_BSM)->Range(100, 10000000);
BENCHMARK_MAIN();