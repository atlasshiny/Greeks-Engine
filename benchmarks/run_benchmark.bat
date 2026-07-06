@echo off
REM Run CPU Benchmark
echo Running CPU Benchmark...
call ..\build\benchmarks\Release\cpu_benchmark.exe --benchmark_out=cpu_results.csv --benchmark_out_format=csv

REM Run GPU Benchmark
echo Running GPU Benchmark...
call ..\build\benchmarks\Release\gpu_benchmark.exe --benchmark_out=gpu_results.csv --benchmark_out_format=csv

echo Benchmarks complete. Results saved to cpu_results.csv and gpu_results.csv.
pause