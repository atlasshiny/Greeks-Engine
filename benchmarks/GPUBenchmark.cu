#include "models/BSMModel.hpp"
#include "setup.hpp"
#include "gpu/GreeksKernel.cuh"
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

static void BM_GPU_BSM(benchmark::State& state) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // WARM-UP BEFORE BENCHMARKING
    const int warmup_size = 1000;
    BenchmarkBatch data = generateBenchmarkBatch(warmup_size);

    launchGreeksKernel(data.options.data(), data.results.data(), warmup_size);
    cudaDeviceSynchronize(); // Ensure the warmup fully completes

    for (auto _ : state) {
        // Generate test inputs
        int n = 1000;

        BenchmarkBatch data = generateBenchmarkBatch(n); // Generate 1000 test cases

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

}
BENCHMARK(BM_GPU_BSM)->UseManualTime(); //This macro registers the benchmark and forces the use of the cudaEvent time
BENCHMARK_MAIN();