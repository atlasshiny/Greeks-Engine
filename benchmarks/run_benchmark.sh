#!/bin/bash
echo "Running CPU Benchmark..."
../build/benchmarks/cpu_benchmark --benchmark_out=cpu_results.csv --benchmark_out_format=csv

echo "Running GPU Benchmark..."
../build/benchmarks/gpu_benchmark --benchmark_out=gpu_results.csv --benchmark_out_format=csv

echo "Running PINN Benchmark..."
../build/benchmarks/pinn_benchmark --benchmark_out=pinn_results.csv --benchmark_out_format=csv

echo "Benchmarks complete. Results saved to cpu_results.csv, gpu_results.csv, and pinn_results.csv"