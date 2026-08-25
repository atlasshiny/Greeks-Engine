# Benchmark Executable
if(CUDAToolkit_FOUND)
    add_executable(gpu_benchmark
        GPUBenchmark.cu
    )

    target_include_directories(gpu_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR})
    target_link_libraries(gpu_benchmark PRIVATE
        gpu_kernels 
        benchmark::benchmark_main
        CUDA::cudart
        CUDA::curand
    )

    set_target_properties(gpu_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()

add_executable(cpu_benchmark
    CPUBenchmark.cpp
)

target_include_directories(cpu_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cpu_benchmark PRIVATE benchmark::benchmark_main)
