#include "models/BSMModel.hpp"
#include "setup.hpp"
#include "gpu/BSMKernel.cuh"
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

static void BM_GPU_BSM(benchmark::State& state) {
    cudaEvent_t start_compute, stop_compute, start_copy, stop_copy;
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);
    cudaEventCreate(&start_copy);
    cudaEventCreate(&stop_copy);

    int n = static_cast<int>(state.range()); // Get the number of test cases from the benchmark state

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    BenchmarkBatch warmup_data = generateBenchmarkBatch(warmup_size);

    launchGreeksKernel(warmup_data.options.data(), warmup_data.results.data(), warmup_size);
    cudaDeviceSynchronize(); // Ensure the warmup fully completes

    // Assign the memory outside the lauchGreeksKernel method to benchmark JUST the calculations

    Option *d_options;
    Greeks *d_results;

    // Allocate memory on GPU
    cudaMalloc(&d_options, n * sizeof(Option));
    cudaMalloc(&d_results, n * sizeof(Greeks));

    // Generate test inputs (Still need to be copied to GPU)
    BenchmarkBatch data = generateBenchmarkBatch(n); // Generate n test cases

    for (auto _ : state) {
        // Record copy start
        cudaEventRecord(start_copy, 0);

        // Copy memory from RAM to VRAM
        cudaMemcpy(d_options, data.options.data(), n * sizeof(Option), cudaMemcpyHostToDevice);
        
        // Record copy stop
        cudaEventRecord(stop_copy, 0);
        cudaEventSynchronize(stop_copy);

        // Find total elapsed time
        float copy_ms = 0;
        cudaEventElapsedTime(&copy_ms, start_copy, stop_copy);

        // Calculate grid dimensions
        int threadsPerBlock = 256;
        int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

        // Record CUDA start
        cudaDeviceSynchronize();
        cudaEventRecord(start_compute, 0);

        // Run the raw kernel without the bridge function
        computeGreeksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_results, n);
        
        // Ensure the calculations are done before stopping the time
        cudaDeviceSynchronize();

        // Record CUDA stop
        cudaEventRecord(stop_compute, 0);
        cudaEventSynchronize(stop_compute);

        // Find total elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_compute, stop_compute);

        // Give time to Google Benchmark
        state.SetIterationTime(milliseconds / 1000.0);

        // Add cudaMemcpy time to the output
        state.counters["Memcpy_GPU_ms"] = copy_ms;

    }

    cudaEventDestroy(start_compute);
    cudaEventDestroy(stop_compute);

    cudaFree(d_options);
    cudaFree(d_results);
}

//This macro registers the benchmark, sets the range for the data, and forces the use of the cudaEvent time
BENCHMARK(BM_GPU_BSM)->Range(100, 10000000)->UseManualTime(); 
BENCHMARK_MAIN();