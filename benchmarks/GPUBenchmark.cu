#include "models/BSMModel.hpp"
#include "setup.hpp"
#include "gpu/BSMKernel.cuh"
#include "gpu/BinomialTreeKernel.cuh"
#include <benchmark/benchmark.h>
#include <cuda_runtime.h>

// BSM BENCHMARKS

static void BM_GPU_BSM_Greeks(benchmark::State& state) {
    cudaEvent_t start_compute, stop_compute, start_copy, stop_copy;
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);
    cudaEventCreate(&start_copy);
    cudaEventCreate(&stop_copy);

    int n = static_cast<int>(state.range()); // Get the number of test cases from the benchmark state

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    BenchmarkBatch warmup_data = generateBenchmarkBatch(warmup_size);

    launchBSMGreeksKernel(warmup_data.options.data(), warmup_data.mktparams.data(), warmup_data.greek_results.data(), warmup_size);

    // Assign the memory outside the lauchGreeksKernel method to benchmark JUST the calculations
    Option *h_options;
    MarketParams *h_mktparams;
    Greeks *h_results;

    cudaHostAlloc((void**)&h_options, n * sizeof(Option), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_mktparams, n * sizeof(MarketParams), cudaHostAllocDefault);
    cudaHostAlloc((void**)&h_results, n * sizeof(Greeks), cudaHostAllocDefault);

    Option *d_options;
    MarketParams *d_mktparams;
    Greeks *d_results;

    // Allocate memory on GPU
    cudaMalloc(&d_options, n * sizeof(Option));
    cudaMalloc(&d_mktparams, n * sizeof(MarketParams));
    cudaMalloc(&d_results, n * sizeof(Greeks));

    // Generate test inputs
    BenchmarkBatch data = generateBenchmarkBatch(n); // Generate n test cases

    // Copy generated test inputs to the pinned memory
    memcpy(h_options, data.options.data(), n * sizeof(Option));
    memcpy(h_mktparams, data.mktparams.data(), n * sizeof(MarketParams));
    memcpy(h_results, data.greek_results.data(), n * sizeof(Greeks));

    for (auto _ : state) {
        // Record copy start
        cudaEventRecord(start_copy, 0);

        // Copy memory from RAM to VRAM
        cudaMemcpy(d_options, h_options, n * sizeof(Option), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mktparams, h_mktparams, n *sizeof(MarketParams), cudaMemcpyHostToDevice);

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
        computeBSMGreeksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_mktparams, d_results, n);
        
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

    // Free host memory
    cudaFreeHost(h_options);
    cudaFreeHost(h_mktparams);
    cudaFreeHost(h_results);

    // Free device memory
    cudaFree(d_options);
    cudaFree(d_mktparams);
    cudaFree(d_results);
}
//This macro registers the benchmark, sets the range for the data, and forces the use of the cudaEvent time
BENCHMARK(BM_GPU_BSM_Greeks)->Range(100, 10000000)->UseManualTime(); // Benchmark for 100 to 10,000,000 options

// BINOMIAL TREE BENCHMARKS

static void BM_GPU_BinomialTree_Greeks(benchmark::State& state) {
    cudaEvent_t start_compute, stop_compute, start_copy, stop_copy;
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);
    cudaEventCreate(&start_copy);
    cudaEventCreate(&stop_copy);

    int n_options = static_cast<int>(state.range()); // Get the number of test cases from the benchmark state
    int n_steps = 200; // Number of steps in the binomial tree

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    constexpr int warmup_step = 10;
    BenchmarkBatch warmup_data = generateBenchmarkBatch(warmup_size);

    launchBinomialGreeksKernel(warmup_data.options.data(), warmup_data.mktparams.data(), warmup_data.greek_results.data(), 0.001, warmup_step, warmup_size);
    cudaDeviceSynchronize(); // Ensure the warmup fully completes

    // Assign the memory outside the lauchGreeksKernel method to benchmark JUST the calculations

    Option *d_options;
    MarketParams *d_mktparams;
    double *d_buffer;
    Greeks *d_results;

    // Allocate memory on GPU
    cudaMalloc(&d_options, n_options * sizeof(Option));
    cudaMalloc(&d_mktparams, n_options * sizeof(MarketParams));
    cudaMalloc(&d_buffer, n_options * (n_steps + 1) * sizeof(double));
    cudaMalloc(&d_results, n_options * sizeof(Greeks));

    // Generate test inputs (Still need to be copied to GPU)
    BenchmarkBatch data = generateBenchmarkBatch(n_options); // Generate n test cases

    for (auto _ : state) {
        // Record copy start
        cudaEventRecord(start_copy, 0);

        // Copy memory from RAM to VRAM
        cudaMemcpy(d_options, data.options.data(), n_options * sizeof(Option), cudaMemcpyHostToDevice);
        cudaMemcpy(d_mktparams, data.mktparams.data(), n_options *sizeof(MarketParams), cudaMemcpyHostToDevice);

        // Record copy stop
        cudaEventRecord(stop_copy, 0);
        cudaEventSynchronize(stop_copy);

        // Find total elapsed time
        float copy_ms = 0;
        cudaEventElapsedTime(&copy_ms, start_copy, stop_copy);

        // Calculate grid dimensions
        int threadsPerBlock = 256;
        int blocksPerGrid = (n_options + threadsPerBlock - 1) / threadsPerBlock;

        // Record CUDA start
        cudaDeviceSynchronize();
        cudaEventRecord(start_compute, 0);

        // Run the raw kernel without the bridge function
        computeBinomialGreeksKernel<<<blocksPerGrid, threadsPerBlock>>>(d_options, d_mktparams, d_results, 0.001, n_steps, n_options, d_buffer);
        
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
    cudaFree(d_mktparams);
    cudaFree(d_buffer);
    cudaFree(d_results);
}
BENCHMARK(BM_GPU_BinomialTree_Greeks)->Range(100, 100000)->UseManualTime(); // Benchmark for 100 to 100,000 options

BENCHMARK_MAIN();