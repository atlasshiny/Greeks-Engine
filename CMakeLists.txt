cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0135 NEW)

project(GreeksEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_ARCHITECTURES native)

    add_compile_definitions(USE_CUDA)

    message("CUDA found; Building GPU and CPU")
else()
    message("No CUDA found; Building CPU only")
endif()

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest
    GIT_TAG v1.17.0
)

FetchContent_Declare(
    googlebenchmark
    GIT_REPOSITORY https://github.com/google/benchmark
    GIT_TAG v1.9.5
)

FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11
    GIT_TAG v3.1.0
)

# Disable Google Benchmark testing for faster build times
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(
    googletest 
    googlebenchmark
    pybind11
)

# GLOBAL PYTHON & TORCH PATH RESOLUTION
find_package(Python COMPONENTS Interpreter REQUIRED)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
)
if (EXISTS "${TORCH_CMAKE_PREFIX}")
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
endif()

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch, os; print(os.path.normcase(os.path.join(os.path.dirname(torch.__file__), 'lib')))"
    OUTPUT_VARIABLE TORCH_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

# Expose discovered paths to subdirectories (like /benchmarks)
set(TORCH_LIB_DIR "${TORCH_LIB_DIR}" CACHE INTERNAL "PyTorch Lib Directory")
set(PYTHON_SITE_PACKAGES "${PYTHON_SITE_PACKAGES}" CACHE INTERNAL "Python Site Packages Directory")

# Find PyTorch Package
find_package(Torch REQUIRED)

# Torch Configs
set(TORCH_CUDA_ARCH_LIST native)

# CUDA Kernel Library Creation
if(CUDAToolkit_FOUND)
    add_library(gpu_kernels STATIC
        src/gpu/BSMKernel.cu
        src/gpu/BinomialTreeKernel.cu
        src/gpu/MonteCarloKernel.cu
    )

    target_include_directories(gpu_kernels PRIVATE include)

    set_target_properties(gpu_kernels PROPERTIES 
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
endif()

# CPU Executable
add_executable(GreeksEngineCPU 
    src/main.cpp
)
target_include_directories(GreeksEngineCPU PRIVATE include)

# GPU Executable
if(CUDAToolkit_FOUND)
    add_executable(GreeksEngineGPU
        src/main.cu
    )

    set_target_properties(GreeksEngineGPU PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_LINK_PHASE ON
    )

    target_include_directories(GreeksEngineGPU PRIVATE include)
    target_link_libraries(GreeksEngineGPU PRIVATE gpu_kernels CUDA::cudart)
endif()

# PyTorch C++ Executable
add_executable(GreeksEnginePINN
    src/PINN.cpp
)
target_link_libraries(GreeksEnginePINN PRIVATE ${TORCH_LIBRARIES})
target_include_directories(GreeksEnginePINN PRIVATE 
    include 
    ${TORCH_INCLUDE_DIRS}
)


# Remove Torch compiler warnings
if (MSVC)
    # Target-specific suppression
    target_compile_options(GreeksEnginePINN PRIVATE /wd4267)
endif()

# AUTOMATED WINDOWS RUNTIME DEPENDENCY DEPLOYMENT FOR MAIN BINARY
if (WIN32)
    # Copy core PyTorch libraries (torch_cuda.dll, c10.dll, etc.)
    add_custom_command(TARGET GreeksEnginePINN POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${TORCH_LIB_DIR}"
            "$<TARGET_FILE_DIR:GreeksEnginePINN>"
        COMMENT "Deploying PyTorch runtime DLLs..."
    )

    # Set Visual Studio Debugger PATH setup for PyTorch and CUDA runtime DLLs
    set_property(TARGET GreeksEnginePINN PROPERTY
        VS_DEBUGGER_ENVIRONMENT "PATH=${TORCH_LIB_DIR};${PYTHON_SITE_PACKAGES}/nvidia/cuda_runtime/bin;${PYTHON_SITE_PACKAGES}/nvidia/cublas/bin;${CUDAToolkit_BIN_DIR};%PATH%"
    )
endif()

# Python Module Bindings
pybind11_add_module(GreeksEngine bindings/bindings.cpp)
target_include_directories(GreeksEngine PRIVATE include)

if(CUDAToolkit_FOUND)
    target_link_libraries(GreeksEngine PRIVATE gpu_kernels CUDA::cudart)
endif()

# Tests
enable_testing()

add_executable(GreeksEngineTests
    tests/bsm_tests.cpp
    tests/binomial_tree_tests.cpp
    tests/monte_carlo_tests.cpp
)

target_include_directories(GreeksEngineTests PRIVATE include)

target_link_libraries(GreeksEngineTests PRIVATE
    GTest::gtest_main   
)

include(GoogleTest)
gtest_discover_tests(GreeksEngineTests)

# Benchmarking CMakeList
add_subdirectory(benchmarks)