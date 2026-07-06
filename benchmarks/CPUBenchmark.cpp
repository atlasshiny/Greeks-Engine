#include "models/BSMModel.hpp"
#include "setup.hpp"
#include <benchmark/benchmark.h>

static void BM_CPU_BSM(benchmark::State& state) {
    for (auto _ : state) {
        // Generate test inputs
        BenchmarkBatch data = generateBenchmarkBatch(1000); // Generate 1000 test cases

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
BENCHMARK(BM_CPU_BSM); // This macro registers the test
BENCHMARK_MAIN();