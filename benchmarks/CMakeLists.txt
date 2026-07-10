# Benchmark Executable
if(CUDAToolkit_FOUND)
    add_executable(gpu_benchmark
        ../src/gpu/BSMKernel.cu # consider finding a better linking solution in the future
        ../src/gpu/BinomialTreeKernel.cu # consider finding a better linking solution in the future
        GPUBenchmark.cu
    )

    target_include_directories(gpu_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(gpu_benchmark PRIVATE benchmark::benchmark_main)
endif()

add_executable(cpu_benchmark
    CPUBenchmark.cpp
)

target_include_directories(cpu_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cpu_benchmark PRIVATE benchmark::benchmark_main)
