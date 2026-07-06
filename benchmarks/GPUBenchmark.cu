#include "models/BSMModel.hpp"
#include "setup.hpp"
#include "gpu/GreeksKernel.cuh"
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

static void BM_GPU_BSM(benchmark::State& state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int n = static_cast<int>(state.range()); // Get the number of test cases from the benchmark state

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    BenchmarkBatch data = generateBenchmarkBatch(warmup_size);

    launchGreeksKernel(data.options.data(), data.results.data(), warmup_size);
    cudaDeviceSynchronize(); // Ensure the warmup fully completes

    for (auto _ : state) {
        // Generate test inputs
        BenchmarkBatch data = generateBenchmarkBatch(n); // Generate n test cases

        // Record CUDA start
        cudaDeviceSynchronize();
        cudaEventRecord(start, 0);

        launchGreeksKernel(data.options.data(), data.results.data(), n);

        // Record CUDA stop
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        // Find total elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        // Give time to Google Benchmark
        state.SetIterationTime(milliseconds / 1000.0);
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

}//This macro registers the benchmark, sets the range for the data, and forces the use of the cudaEvent time
BENCHMARK(BM_GPU_BSM)->Range(100, 10000000)->UseManualTime(); 
BENCHMARK_MAIN();