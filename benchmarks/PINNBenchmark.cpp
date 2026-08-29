#include <benchmark/benchmark.h>
#include <torch/torch.h>
#include "setup.hpp"
#include "models/BSMModel.hpp"
#include "models/PhysicsInformedNN.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include "gpu/BSMKernel.cuh"
#endif

// CPU BENCHMARKS (Always compiled to allow CPU vs GPU comparative plotting)
static void BM_CPU_PINN_TargetGen(benchmark::State& state) {
    int n = static_cast<int>(state.range());
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    auto S = torch::rand({n, 1}, options) * 100.0 + 50.0;
    auto K = torch::rand({n, 1}, options) * 40.0 + 80.0;
    auto T = torch::rand({n, 1}, options) * 1.5 + 0.1;
    auto r = torch::full({n, 1}, 0.05, options);
    auto sigma = torch::full({n, 1}, 0.2, options);
    auto opt_type = torch::zeros({n, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));

    for (auto _ : state) {
        auto targets = PhysicsInformedNN::generate_targets(S, K, T, r, sigma, opt_type);
        benchmark::DoNotOptimize(targets.data_ptr());
    }
}
BENCHMARK(BM_CPU_PINN_TargetGen)->Range(100, 100000);

static void BM_CPU_PINN_ComputeLoss(benchmark::State& state) {
    int n = static_cast<int>(state.range());
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    PhysicsInformedNN model;
    model.to(torch::kCPU);

    auto S = torch::rand({n, 1}, options) * 100.0 + 50.0;
    auto K = torch::rand({n, 1}, options) * 40.0 + 80.0;
    auto T = torch::rand({n, 1}, options) * 1.5 + 0.1;
    auto r = torch::full({n, 1}, 0.05, options);
    auto sigma = torch::full({n, 1}, 0.2, options);
    auto opt_type = torch::zeros({n, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU));
    auto targets = PhysicsInformedNN::generate_targets(S, K, T, r, sigma, opt_type);

    for (auto _ : state) {
        auto loss = model.compute_loss(S, K, T, r, sigma, opt_type, targets);
        benchmark::DoNotOptimize(loss.data_ptr());
    }
}
BENCHMARK(BM_CPU_PINN_ComputeLoss)->Range(100, 100000);

static void BM_CPU_PINN_EvaluateGreeks(benchmark::State& state) {
    int n = static_cast<int>(state.range());
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCPU);

    PhysicsInformedNN model;
    model.to(torch::kCPU);

    auto S = torch::rand({n, 1}, options) * 100.0 + 50.0;
    auto K = torch::rand({n, 1}, options) * 40.0 + 80.0;
    auto T = torch::rand({n, 1}, options) * 1.5 + 0.1;
    auto r = torch::full({n, 1}, 0.05, options);
    auto sigma = torch::full({n, 1}, 0.2, options);

    for (auto _ : state) {
        auto greeks = model.evaluate_greeks(S, K, T, r, sigma);
        benchmark::DoNotOptimize(&greeks);
    }
}
BENCHMARK(BM_CPU_PINN_EvaluateGreeks)->Range(100, 100000);


// GPU BENCHMARKS (Compiled only when NVCC CUDA compiler is active)
#ifdef __CUDACC__

static void BM_GPU_PINN_TargetGen(benchmark::State& state) {
    cudaEvent_t start_compute, stop_compute;
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);

    int n = static_cast<int>(state.range());

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    auto warmup_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto S_warm = torch::rand({warmup_size, 1}, warmup_options) * 100.0 + 50.0;
    auto K_warm = torch::rand({warmup_size, 1}, warmup_options) * 40.0 + 80.0;
    auto T_warm = torch::rand({warmup_size, 1}, warmup_options) * 1.5 + 0.1;
    auto r_warm = torch::full({warmup_size, 1}, 0.05, warmup_options);
    auto sig_warm = torch::full({warmup_size, 1}, 0.2, warmup_options);
    auto opt_warm = torch::zeros({warmup_size, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    PhysicsInformedNN::generate_targets(S_warm, K_warm, T_warm, r_warm, sig_warm, opt_warm);
    cudaDeviceSynchronize();

    // Allocate GPU Tensors directly in VRAM
    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto S = torch::rand({n, 1}, options) * 100.0 + 50.0;
    auto K = torch::rand({n, 1}, options) * 40.0 + 80.0;
    auto T = torch::rand({n, 1}, options) * 1.5 + 0.1;
    auto r = torch::full({n, 1}, 0.05, options);
    auto sigma = torch::full({n, 1}, 0.2, options);
    auto opt_type = torch::zeros({n, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));

    for (auto _ : state) {
        // Record CUDA start
        cudaDeviceSynchronize();
        cudaEventRecord(start_compute, 0);

        auto targets = PhysicsInformedNN::generate_targets(S, K, T, r, sigma, opt_type);

        // Ensure calculations are complete before stopping time
        cudaDeviceSynchronize();
        cudaEventRecord(stop_compute, 0);
        cudaEventSynchronize(stop_compute);

        // Find total elapsed time
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_compute, stop_compute);

        // Give time to Google Benchmark
        state.SetIterationTime(milliseconds / 1000.0);
        state.counters["Memcpy_GPU_ms"] = 0.0; // Zero-copy tensor pipeline
    }

    cudaEventDestroy(start_compute);
    cudaEventDestroy(stop_compute);
}
BENCHMARK(BM_GPU_PINN_TargetGen)->Range(100, 10000000)->UseManualTime();

static void BM_GPU_PINN_ComputeLoss(benchmark::State& state) {
    cudaEvent_t start_compute, stop_compute;
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);

    int n = static_cast<int>(state.range());

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    auto warmup_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    PhysicsInformedNN warmup_model;
    warmup_model.to(torch::kCUDA);

    auto S_warm = torch::rand({warmup_size, 1}, warmup_options) * 100.0 + 50.0;
    auto K_warm = torch::rand({warmup_size, 1}, warmup_options) * 40.0 + 80.0;
    auto T_warm = torch::rand({warmup_size, 1}, warmup_options) * 1.5 + 0.1;
    auto r_warm = torch::full({warmup_size, 1}, 0.05, warmup_options);
    auto sig_warm = torch::full({warmup_size, 1}, 0.2, warmup_options);
    auto opt_warm = torch::zeros({warmup_size, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto targets_warm = PhysicsInformedNN::generate_targets(S_warm, K_warm, T_warm, r_warm, sig_warm, opt_warm);

    warmup_model.compute_loss(S_warm, K_warm, T_warm, r_warm, sig_warm, opt_warm, targets_warm);
    cudaDeviceSynchronize();

    // Allocate Model and Input Tensors
    PhysicsInformedNN model;
    model.to(torch::kCUDA);

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto S = torch::rand({n, 1}, options) * 100.0 + 50.0;
    auto K = torch::rand({n, 1}, options) * 40.0 + 80.0;
    auto T = torch::rand({n, 1}, options) * 1.5 + 0.1;
    auto r = torch::full({n, 1}, 0.05, options);
    auto sigma = torch::full({n, 1}, 0.2, options);
    auto opt_type = torch::zeros({n, 1}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA));
    auto targets = PhysicsInformedNN::generate_targets(S, K, T, r, sigma, opt_type);

    for (auto _ : state) {
        cudaDeviceSynchronize();
        cudaEventRecord(start_compute, 0);

        auto loss = model.compute_loss(S, K, T, r, sigma, opt_type, targets);

        cudaDeviceSynchronize();
        cudaEventRecord(stop_compute, 0);
        cudaEventSynchronize(stop_compute);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_compute, stop_compute);

        state.SetIterationTime(milliseconds / 1000.0);
        state.counters["Memcpy_GPU_ms"] = 0.0;
    }

    cudaEventDestroy(start_compute);
    cudaEventDestroy(stop_compute);
}
BENCHMARK(BM_GPU_PINN_ComputeLoss)->Range(100, 1000000)->UseManualTime();

static void BM_GPU_PINN_EvaluateGreeks(benchmark::State& state) {
    cudaEvent_t start_compute, stop_compute;
    cudaEventCreate(&start_compute);
    cudaEventCreate(&stop_compute);

    int n = static_cast<int>(state.range());

    // WARM-UP BEFORE BENCHMARKING
    constexpr int warmup_size = 1000;
    auto warmup_options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    PhysicsInformedNN warmup_model;
    warmup_model.to(torch::kCUDA);

    auto S_warm = torch::rand({warmup_size, 1}, warmup_options) * 100.0 + 50.0;
    auto K_warm = torch::rand({warmup_size, 1}, warmup_options) * 40.0 + 80.0;
    auto T_warm = torch::rand({warmup_size, 1}, warmup_options) * 1.5 + 0.1;
    auto r_warm = torch::full({warmup_size, 1}, 0.05, warmup_options);
    auto sig_warm = torch::full({warmup_size, 1}, 0.2, warmup_options);

    warmup_model.evaluate_greeks(S_warm, K_warm, T_warm, r_warm, sig_warm);
    cudaDeviceSynchronize();

    PhysicsInformedNN model;
    model.to(torch::kCUDA);

    auto options = torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA);
    auto S = torch::rand({n, 1}, options) * 100.0 + 50.0;
    auto K = torch::rand({n, 1}, options) * 40.0 + 80.0;
    auto T = torch::rand({n, 1}, options) * 1.5 + 0.1;
    auto r = torch::full({n, 1}, 0.05, options);
    auto sigma = torch::full({n, 1}, 0.2, options);

    for (auto _ : state) {
        cudaDeviceSynchronize();
        cudaEventRecord(start_compute, 0);

        auto greeks = model.evaluate_greeks(S, K, T, r, sigma);

        cudaDeviceSynchronize();
        cudaEventRecord(stop_compute, 0);
        cudaEventSynchronize(stop_compute);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start_compute, stop_compute);

        state.SetIterationTime(milliseconds / 1000.0);
        state.counters["Memcpy_GPU_ms"] = 0.0;
    }

    cudaEventDestroy(start_compute);
    cudaEventDestroy(stop_compute);
}
BENCHMARK(BM_GPU_PINN_EvaluateGreeks)->Range(100, 1000000)->UseManualTime();

#endif

BENCHMARK_MAIN();