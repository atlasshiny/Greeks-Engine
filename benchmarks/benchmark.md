# Benchmarks

Full CPU/GPU benchmark methodology, raw output, and performance analysis for the Greeks Engine. See [README.md](README.md) for a project overview.

**Hardware:** Intel i7-10700KF, NVIDIA RTX 5060 (8GB), Windows 11

---

## Methodology

**GPU timing uses `cudaEvent_t`, not `std::chrono`.** Host-side timers include CPU-GPU synchronization overhead that inflates GPU measurements. `cudaEvent_t` records timestamps directly on the GPU's command stream, giving accurate kernel-only timing.

**Compute and memory transfer are measured separately.** Each GPU benchmark uses two independent `cudaEvent_t` pairs — one wrapping the kernel launch and one wrapping the host-device `cudaMemcpy` calls. The `Time` column reports compute-only kernel time; the `Memcpy_GPU_ms` counter reports host-device transfer time for the same iteration. This separation is what makes it possible to say *where* GPU time is actually going, rather than just reporting a single opaque number (though using thorough profiling via Nsight is more precise).

**A warmup pass precedes every timed run.** Both CPU and GPU benchmarks run once, untimed, before entering the Google Benchmark measurement loop. This excludes CUDA driver initialization and cache-cold effects from the reported numbers.

**Batch data is generated once, outside the timing loop.** Vector allocation and option generation happen before the benchmark loop starts, so the reported time reflects pricing/Greeks computation only, not setup cost.

---

## CPU Benchmark

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

---

## GPU Benchmark

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

`Time` is GPU **compute-only** time (kernel execution). `Memcpy_GPU_ms` is host-device transfer time for the same batch, measured independently. Total GPU wall time for a batch is the sum of the two.

*Note: `BinomialTreeModel` is only benchmarked up to 100,000 options due to long computational times.*
 
---

## Computed Speedup (CPU time ÷ total GPU time)

Total GPU time = compute (`Time`) + `Memcpy_GPU_ms` (converted to ns).

### BSM — Greeks

| Batch size | CPU (ns) | GPU compute (ns) | GPU memcpy (ns) | GPU total (ns) | Speedup |
|---|---|---|---|---|---|
| 100 | 3,497 | 33,645 | 28,032 | 61,677 | 0.06x |
| 512 | 18,352 | 39,569 | 34,080 | 73,649 | 0.25x |
| 4,096 | 142,430 | 33,495 | 96,480 | 129,975 | 1.10x |
| 32,768 | 1,144,810 | 88,988 | 335,744 | 424,732 | 2.70x |
| 262,144 | 9,452,079 | 477,384 | 2,252,030 | 2,729,414 | 3.46x |
| 2,097,152 | 75,509,156 | 3,530,006 | 17,633,000 | 21,163,006 | 3.57x |
| 10,000,000 | 383,898,100 | 16,554,763 | 82,501,800 | 99,056,563 | 3.88x |

GPU is *slower* than CPU below ~4,096 options (fixed kernel launch and transfer overhead dominates at small batch sizes). Speedup plateaus around **3.9x** at large batches, well below the raw compute speedup, because full Greeks computation (5 closed-form values per option) is transfer-bound: memcpy time is ~5x larger than compute time at 10M options.

### Binomial Tree — Greeks

| Batch size | CPU (ns) | GPU compute (ns) | GPU memcpy (ns) | GPU total (ns) | Speedup |
|---|---|---|---|---|---|
| 100 | 23,317,432 | 46,676,670 | 107,808 | 46,784,478 | 0.50x |
| 512 | 122,478,083 | 71,892,400 | 31,040 | 71,923,440 | 1.70x |
| 4,096 | 962,689,900 | 76,812,699 | 139,200 | 76,951,899 | 12.51x |
| 32,768 | 7,695,476,000 | 505,342,712 | 475,680 | 505,818,392 | 15.22x |
| 100,000 | 23,672,000,000 | 1,276,642,578 | 1,321,890 | 1,277,964,468 | **18.53x** |

Binomial Tree GPU speedup grows faster and reaches a higher ceiling than BSM's (18.5x vs. 3.9x at comparable scale). Memcpy is a negligible fraction of total time at every batch size shown; the workload is compute-bound, not transfer-bound.

---

## Architectural Takeaway

| Model | Bottleneck | Why |
|---|---|---|
| BSM | Memory transfer | Closed-form formulas with minimal work per thread, so H2D/D2H transfer dominates total time at scale |
| Binomial Tree | Compute | Large tree evaluation per thread, repeated 5x for finite-difference Greeks. The transfer cost is negligible by comparison |

This is corroborated by an NVIDIA Nsight Systems & Compute profile of both kernels, in addition to the benchmark numbers above. Further GPU optimization for BSM should target reducing transfer volume (like batching multiple option batches per transfer, or computing Greeks in-place without round-tripping intermediate values), while further optimization for the Binomial Tree should target compute (like using shared memory usage for the backward sweep or fewer finite-difference evaluations per Greek).

---

## Plots

### BSM

![BSM Performance Trend](diagrams/BSM/performance_pointplot.png)
![BSM CPU Performance](diagrams/BSM/cpu_performance.png)
![BSM Speedup Ratio](diagrams/BSM/speedup_ratio.png)
![BSM GPU Compute vs Memcpy Breakdown](diagrams/BSM/gpu_breakdown_plot.png)
![BSM GPU Memcpy Time](diagrams/BSM/gpu_memcpy.png)
![BSM CDF of Execution Times](diagrams/BSM/cdf_plot.png)

### Binomial Tree

![Binomial Tree Performance Trend](diagrams/BinomialTree/performance_pointplot.png)
![Binomial Tree CPU Performance](diagrams/BinomialTree/cpu_performance.png)
![Binomial Tree Speedup Ratio](diagrams/BinomialTree/speedup_ratio.png)
![Binomial Tree GPU Compute vs Memcpy Breakdown](diagrams/BinomialTree/gpu_breakdown_plot.png)
![Binomial Tree GPU Memcpy Time](diagrams/BinomialTree/gpu_memcpy.png)
![Binomial Tree CDF of Execution Times](diagrams/BinomialTree/cdf_plot.png)

*Note: the red line in the speedup ratio plots marks the 1:1 CPU/GPU ratio — anything above it is a real GPU speedup, anything below means the CPU was faster.*

---

## Reproducing These Results

```bash
cd benchmarks

# Linux/macOS
./run_benchmark.sh

# Windows
run_benchmark.bat

# Generate plots from the resulting CSVs
pip install -r ../requirements.txt
python plot.py
```

`run_benchmark.sh`/`.bat` runs both `cpu_benchmark` and `gpu_benchmark`, writing `cpu_results.csv` and `gpu_results.csv`. `plot.py` reads both files, tags each row by model (BSM vs. BinomialTree), and generates the plots above into per-model subdirectories under `diagrams/`.

Absolute numbers will differ by hardware. The relative story of BSM memory-bound, Binomial Tree compute-bound, both showing a CPU/GPU crossover point below which the GPU is not worth using should hold across most consumer GPUs.