#include "models/BSMModel.hpp"
#include "setup.hpp"
#include <benchmark/benchmark.h>

static void BM_CPU_BSM(benchmark::State& state) {

    int n = static_cast<int>(state.range()); // Get the number of test cases from the benchmark state

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    BenchmarkBatch warmup_data = generateBenchmarkBatch(warmup_size);

    for (size_t i = 0; i < warmup_data.options.size(); ++i) {
        BSMModel model(warmup_data.mktparams[i].S, warmup_data.options[i].K, warmup_data.options[i].T, warmup_data.mktparams[i].r, warmup_data.mktparams[i].sigma);
        model.calculateGreeks(warmup_data.options[i].type); // Calculate the Greeks (not stored, just for warm-up)
    }

    // Generate test inputs
    BenchmarkBatch data = generateBenchmarkBatch(n); // Generate n test cases

    for (auto _ : state) {
        // Run the BSM model on each option and store the results
        for (size_t i = 0; i < data.options.size(); ++i) {
            BSMModel model(data.mktparams[i].S, data.options[i].K, data.options[i].T, data.mktparams[i].r, data.mktparams[i].sigma);

            model.calculateGreeks(data.options[i].type); // Calculate the Greeks (not stored, just for benchmarking)
        }
    }
}

BENCHMARK(BM_CPU_BSM)->Range(100, 10000000);
BENCHMARK_MAIN();