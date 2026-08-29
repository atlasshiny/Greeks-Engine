@echo off
REM Run CPU Benchmark
echo Running CPU Benchmark...
call ..\build\benchmarks\Release\cpu_benchmark.exe --benchmark_out=cpu_results.csv --benchmark_out_format=csv

REM Run GPU Benchmark
echo Running GPU Benchmark...
call ..\build\benchmarks\Release\gpu_benchmark.exe --benchmark_out=gpu_results.csv --benchmark_out_format=csv

REM Run PINN Benchmark
echo Running PINN Benchmark...
call ..\build\benchmarks\Release\pinn_benchmark.exe --benchmark_out=pinn_results.csv --benchmark_out_format=csv

echo Benchmarks complete. Results saved to cpu_results.csv, gpu_results.csv, and pinn_results.csv.
pause