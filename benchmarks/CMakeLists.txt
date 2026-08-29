# Benchmark Executable

# GPU Benchmark
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

# CPU Benchmark
add_executable(cpu_benchmark
    CPUBenchmark.cpp
)

target_include_directories(cpu_benchmark PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../include ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(cpu_benchmark PRIVATE benchmark::benchmark_main)

# PINN Benchmark
add_executable(pinn_benchmark
    PINNBenchmark.cpp
)

target_include_directories(pinn_benchmark PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/../include 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

if(CUDAToolkit_FOUND)
    # Force NVCC to compile the .cpp file so __CUDACC__ is defined
    set_source_files_properties(PINNBenchmark.cpp PROPERTIES LANGUAGE CUDA)

    target_link_libraries(pinn_benchmark PRIVATE
        gpu_kernels
        benchmark::benchmark_main
        CUDA::cudart
        CUDA::curand
        ${TORCH_LIBRARIES}
    )

    set_target_properties(pinn_benchmark PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    # Enable modern MSVC preprocessor for CCCL compatibility
    if(MSVC)
        target_compile_options(pinn_benchmark PRIVATE
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>
            $<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>
        )
    endif()

else()
    target_link_libraries(pinn_benchmark PRIVATE
        benchmark::benchmark_main
        ${TORCH_LIBRARIES}
    )
endif()