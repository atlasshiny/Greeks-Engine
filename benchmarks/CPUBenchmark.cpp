#include "models/BSMModel.hpp"
#include "models/BinomialTreeModel.hpp"
#include "setup.hpp"
#include <benchmark/benchmark.h>

// BSM BENCHMARKS

static void BM_CPU_BSM_Greeks(benchmark::State& state) {

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
BENCHMARK(BM_CPU_BSM_Greeks)->Range(100, 10000000); // Benchmark for 100 to 10,000,000 options

// BINOMIAL TREE BENCHMARKS

static void BM_CPU_BinomialTreeGreeks(benchmark::State& state) {

    int n_options = static_cast<int>(state.range()); // Get the number of test cases from the benchmark state
    int n_steps = 200; // Number of steps in the binomial tree

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    constexpr int warmup_steps = 50; // Use fewer steps for warm-up to speed up the process
    BenchmarkBatch warmup_data = generateBenchmarkBatch(warmup_size);
    std::vector<double> warmup_buffer(warmup_steps + 1); // Buffer size is warmup_steps + 1

    for (size_t i = 0; i < warmup_data.options.size(); ++i) {
        BinomialTreeModel model(warmup_data.mktparams[i].S, warmup_data.options[i].K, warmup_data.options[i].T, warmup_data.mktparams[i].r, warmup_data.mktparams[i].sigma, warmup_data.mktparams[i].q, warmup_steps, warmup_data.options[i].isAmerican);
        model.calculateGreeks(warmup_data.options[i].type, warmup_buffer.data(), 0.001); // Calculate the Greeks (not stored, just for warm-up)
    }

    // Generate test inputs
    BenchmarkBatch data = generateBenchmarkBatch(n_options); // Generate n test cases

    // Create a buffer for the binomial tree calculations to avoid dynamic memory allocation during benchmarking
    std::vector<double> buffer(n_steps + 1); // Buffer size is n_steps + 1

    for (auto _ : state) {
        // Run the Binomial Tree model on each option and store the results
        for (size_t i = 0; i < data.options.size(); ++i) {
            BinomialTreeModel model(data.mktparams[i].S, data.options[i].K, data.options[i].T, data.mktparams[i].r, data.mktparams[i].sigma, data.mktparams[i].q, n_steps, data.options[i].isAmerican);

            model.calculateGreeks(data.options[i].type, buffer.data(), 0.01); // Calculate the Greeks (not stored, just for benchmarking)
        }
    }
}
BENCHMARK(BM_CPU_BinomialTreeGreeks)->Range(100, 100000); // Benchmark for 100 to 100,000 options

BENCHMARK_MAIN();