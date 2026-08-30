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
if(Torch_FOUND)
    add_executable(pinn_benchmark 
        PINNBenchmark.cpp 
    )
    target_include_directories(pinn_benchmark PRIVATE 
        ${PROJECT_SOURCE_DIR}/include
        include
    )
    target_link_libraries(pinn_benchmark PRIVATE ${TORCH_LIBRARIES})

    if(CUDAToolkit_FOUND)
        set_property(TARGET pinn_benchmark PROPERTY CUDA_ARCHITECTURES native)
    endif()
endif()

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
endif()

# WINDOWS DLL RESOLUTION & RUN TARGETS FOR PINN BENCHMARK
if (WIN32)
    # Post-build DLL deployment directly next to pinn_benchmark.exe
    add_custom_command(TARGET pinn_benchmark POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${TORCH_LIB_DIR}"
            "$<TARGET_FILE_DIR:pinn_benchmark>"
        COMMENT "Deploying PyTorch dynamic libraries to benchmarks output directory..."
    )

    # Visual Studio IDE Debugging Path setup
    set_property(TARGET pinn_benchmark PROPERTY
        VS_DEBUGGER_ENVIRONMENT "PATH=${TORCH_LIB_DIR};${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/bin;${PYTHON_SITE_PACKAGES}/nvidia/cublas/bin;${CUDAToolkit_BIN_DIR};%PATH%"
    )

    # Custom Run Target for CLI Execution
    add_custom_target(run_pinn_benchmark
        COMMAND ${CMAKE_COMMAND} -E env 
            "PATH=${TORCH_LIB_DIR};${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/bin;${PYTHON_SITE_PACKAGES}/nvidia/cublas/bin;${CUDAToolkit_BIN_DIR};$ENV{PATH}"
            "$<TARGET_FILE:pinn_benchmark>"
        DEPENDS pinn_benchmark
        WORKING_DIRECTORY "$<TARGET_FILE_DIR:pinn_benchmark>"
        COMMENT "Running pinn_benchmark with runtime CUDA & PyTorch PATH injected..."
    )
endif()