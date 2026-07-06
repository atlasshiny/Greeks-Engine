cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0135 NEW)

project(GreeksEngine LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit QUIET)

if(CUDAToolkit_FOUND)
    enable_language(CUDA)
    set(CMAKE_CUDA_ARCHITECTURES native)

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

FetchContent_MakeAvailable(
    googletest 
    googlebenchmark
)

# Disable Google Benchmark testing for faster build times
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

# CPU Executable
add_executable(GreeksEngineCPU 
    src/main.cpp
)
target_include_directories(GreeksEngineCPU PRIVATE include)

# GPU Executable
if(CUDAToolkit_FOUND)
    add_executable(GreeksEngineGPU
        src/GreeksKernel.cu
        src/main.cu
    )

    target_include_directories(GreeksEngineGPU PRIVATE include)
endif()

# Benchmarking CMakeList
add_subdirectory(benchmarks)

# Tests
enable_testing()

add_executable(GreeksEngineTests
    tests/bsm_tests.cpp
)

target_include_directories(GreeksEngineTests PRIVATE include)

target_link_libraries(GreeksEngineTests PRIVATE
    GTest::gtest_main   
)

include(GoogleTest)
gtest_discover_tests(GreeksEngineTests)