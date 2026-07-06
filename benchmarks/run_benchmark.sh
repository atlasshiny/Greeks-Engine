#!/bin/bash
echo "Running CPU Benchmark..."
../build/benchmarks/Release/cpu_benchmark --benchmark_out=cpu_results.csv --benchmark_out_format=csv

echo "Running GPU Benchmark..."
../build/benchmarks/Release/gpu_benchmark --benchmark_out=gpu_results.csv --benchmark_out_format=csv

echo "Benchmarks complete. Results saved to cpu_results.csv and gpu_results.csv."