# Greeks Engine

![C++](https://img.shields.io/badge/C%2B%2B-20-00599C?style=for-the-badge&logo=cplusplus&logoColor=white)
![CUDA](https://img.shields.io/badge/Language-CUDA-green?style=for-the-badge&logo=nvidia&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux%20%7C%20macOS-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

A GPU-accelerated option pricing library in C++20. Implements the Black-Scholes-Merton model for European options and the Binomial Tree model for American options, computing prices and all five major Greeks (Δ, Γ, ν, θ, ρ) on both CPU and CUDA GPU from a single  header-only codebase.

## Features

- Prices European calls and puts with the Black-Scholes-Merton model.
- Prices European and American options with the binomial tree model.
- Computes the major Greeks exposed by each model.
- Provides CPU and GPU executable entry points from the same codebase.
- Includes GoogleTest unit tests and Google Benchmark benchmarks.
- CMake auto-detects CUDA; builds CPU-only if no GPU is found

## Project Structure

```
Greeks-Engine/
├── include/
│   ├── Greeks.hpp
│   ├── MarketParameters.hpp
│   ├── Option.hpp
│   ├── macros.hpp
│   ├── math/
│   │   ├── normcdf.hpp
│   │   └── normpdf.hpp
│   ├── models/
│   │   ├── BSMModel.hpp
│   │   ├── BinomialTreeModel.hpp
│   │   └── MonteCarloModel.hpp
│   └── gpu/
│       ├── BSMKernel.cuh
│       ├── BinomialTreeKernel.cuh
│       └── error_checking.cuh
├── src/
│   ├── main.cpp
│   ├── main.cu
│   └── gpu/
│       ├── BSMKernel.cu
│       └── BinomialTreeKernel.cu
├── benchmarks/
│   ├── CPUBenchmark.cpp
│   ├── GPUBenchmark.cu
│   ├── plot.py
│   ├── setup.hpp
│   ├── cpu_results.csv
│   └── gpu_results.csv
├── tests/
│   ├── bsm_tests.cpp
│   └── binomial_tree_tests.cpp
└── diagrams/
    ├── BSM/
    └── BinomialTree/
```

## Build

***Requirements:**** C++20 compiler, CMake 3.20+. CUDA Toolkit is optional — the build falls back to CPU-only if not found.

```bash
git clone https://github.com/atlasshiny/Greeks-Engine.git
cd Greeks-Engine
mkdir build && cd build
cmake ..
cmake --build . -j4
```

This produces up to five executables depending on your environment:

| Executable | Description |
|---|---|
| `GreeksEngineCPU` | CPU demo that prints BSM and binomial-tree outputs |
| `GreeksEngineGPU` | CUDA demo that runs the batch GPU implementations |
| `GreeksEngineTests` | GoogleTest suite for the pricing models |
| `cpu_benchmark` | Google Benchmark target for sequential CPU pricing |
| `gpu_benchmark` | Google Benchmark target for CUDA kernel GPU pricing |

## Running

The exact executable path depends on your CMake generator and build configuration. After building, run the demo targets from the build tree.

### CPU Demo

Runs the default BSM and binomial tree examples from [src/main.cpp](src/main.cpp).

### GPU Demo

Runs the batch BSM and binomial tree examples from [src/main.cu](src/main.cu) when CUDA is available.

### Tests

Use CTest or the generated test executable from the build tree.

## Benchmarks
The benchmark sources live in [benchmarks/CPUBenchmark.cpp](benchmarks/CPUBenchmark.cpp) and [benchmarks/GPUBenchmark.cu](benchmarks/GPUBenchmark.cu). The plotting helper is [benchmarks/plot.py](benchmarks/plot.py).

The following results and plots were generated using an Intel i7-10700KF and RTX 5060 (8GB) on Windows 11

### CPU Benchmark
```bash
Running CPU Benchmark...
Running ..\build\benchmarks\Release\cpu_benchmark.exe
Run on (16 X 3792 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 256 KiB (x8)
  L3 Unified 16384 KiB (x1)
---------------------------------------------------------------------------
Benchmark                                 Time             CPU   Iterations
---------------------------------------------------------------------------
BM_CPU_BSM_Greeks/100                  3497 ns         3530 ns       203636
BM_CPU_BSM_Greeks/512                 18352 ns        18415 ns        37333
BM_CPU_BSM_Greeks/4096               142430 ns       144385 ns         4978
BM_CPU_BSM_Greeks/32768             1144810 ns      1147461 ns          640
BM_CPU_BSM_Greeks/262144            9452079 ns      9583333 ns           75
BM_CPU_BSM_Greeks/2097152          75509156 ns     74652778 ns            9
BM_CPU_BSM_Greeks/10000000        383898100 ns    382812500 ns            2
BM_CPU_BinomialTreeGreeks/100      23317432 ns     22879464 ns           28
BM_CPU_BinomialTreeGreeks/512     122478083 ns    122395833 ns            6
BM_CPU_BinomialTreeGreeks/4096    962689900 ns    953125000 ns            1
BM_CPU_BinomialTreeGreeks/32768  7695476000 ns   7656250000 ns            1
BM_CPU_BinomialTreeGreeks/100000 2.3672e+10 ns   2.3594e+10 ns            1
```

### GPU Benchmark

```bash
Running GPU Benchmark...
Running ..\build\benchmarks\Release\gpu_benchmark.exe
Run on (16 X 3792 MHz CPU s)
CPU Caches:
  L1 Data 32 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 256 KiB (x8)
  L3 Unified 16384 KiB (x1)
--------------------------------------------------------------------------------------------------------
Benchmark                                              Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------------
BM_GPU_BSM_Greeks/100/manual_time                  33645 ns        97011 ns        20133 Memcpy_GPU_ms=0.028032
BM_GPU_BSM_Greeks/512/manual_time                  39569 ns       103688 ns        17179 Memcpy_GPU_ms=0.03408
BM_GPU_BSM_Greeks/4096/manual_time                 33495 ns       128029 ns        19893 Memcpy_GPU_ms=0.09648
BM_GPU_BSM_Greeks/32768/manual_time                88988 ns       438603 ns         7873 Memcpy_GPU_ms=0.335744
BM_GPU_BSM_Greeks/262144/manual_time              477384 ns      2761727 ns         1471 Memcpy_GPU_ms=2.25203
BM_GPU_BSM_Greeks/2097152/manual_time            3530006 ns     21024215 ns          191 Memcpy_GPU_ms=17.633
BM_GPU_BSM_Greeks/10000000/manual_time          16554763 ns     98958333 ns           42 Memcpy_GPU_ms=82.5018
BM_GPU_BinomialTree_Greeks/100/manual_time      46676670 ns     45833333 ns           15 Memcpy_GPU_ms=0.107808
BM_GPU_BinomialTree_Greeks/512/manual_time      71892400 ns     71875000 ns           10 Memcpy_GPU_ms=0.03104
BM_GPU_BinomialTree_Greeks/4096/manual_time     76812699 ns     76388889 ns            9 Memcpy_GPU_ms=0.1392
BM_GPU_BinomialTree_Greeks/32768/manual_time   505342712 ns    500000000 ns            1 Memcpy_GPU_ms=0.47568
BM_GPU_BinomialTree_Greeks/100000/manual_time 1276642578 ns   1281250000 ns            1 Memcpy_GPU_ms=1.32189
```

### Benchmark Plots
![Performance Comparison BSM](diagrams/BSM/performance_pointplot.png)
![Performance Comparison BinomialTree](diagrams/BinomialTree/performance_pointplot.png)

## Notes

- `BSMModel` is the most mature path and is used by the main pricing examples and tests.
- `BinomialTreeModel` supports pricing and finite-difference Greeks, but it is more computationally expensive.
- While `BSMModel` is bottlenecked by memory transfers, `BinomialTreeModel` is bottlenecked by expensive calculations (backed by a NVIDIA Nsight Systems & Compute profile of the code as well as the benchmarking results).

## Known Limitations

- **Constant volatility** — BSM assumes σ is fixed; where stochastic volatility models (like Heston) would be more realistic
- **No calibration** — implied volatility solving and surface fitting are not implemented

## Planned Extensions
- Monte Carlo pricing (for exotic options)
- Implied volatility solver (starting with the solver and then potentially moving to a surface)
- Hardening math for edge cases such as div by zero, div by infinity, ect.
- Swapping kernel math functions for intrisic CUDA functions (Note that accuracy could fall in this swap and that why it hasn't happened yet)

## Design Notes

**Why header-only?**
CUDA requires that any function called from a `__device__` context is visible to `nvcc` at compile time. Moving `BSMModel` entirely into a header lets the same class compile for both CPU (`g++`) and GPU (`nvcc`) without code duplication.

**Why `HOST_DEVICE` on every method?**
The `__host__ __device__` qualifiers tell `nvcc` to emit both a CPU and a GPU version of each function. The `#ifdef __CUDACC__` guard makes the macro a no-op under a standard C++ compiler, so the headers remain clean for CPU-only builds.

**Why `cudaEvent_t` for benchmarking?**
Host-side timers (`std::chrono`) include synchronization overhead that inflates GPU measurements. `cudaEvent_t` records timestamps on the GPU's command stream, giving accurate kernel-only timing. The GPU benchmark also separates `cudaMemcpy` time from compute time so both costs are visible independently.

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*. Journal of Political Economy.
- NVIDIA CUDA Programming Guide — [docs.nvidia.com/cuda](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- Google Benchmark — [github.com/google/benchmark](https://github.com/google/benchmark)
