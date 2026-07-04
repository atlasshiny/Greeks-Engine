cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0135 NEW)

project(GreeksEngine LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES native)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(googletest)

# BSM model library
add_library(BSMLib 
    src/models/BSMModel.cpp
)
target_include_directories(BSMLib PUBLIC include)

# Main Executable
add_executable(GreeksEngine 
    src/main.cpp
)
target_include_directories(GreeksEngine PRIVATE include)
target_link_libraries(GreeksEngine PRIVATE BSMLib)

# Tests
enable_testing()

add_executable(GreeksEngineTests
    tests/bsm_tests.cpp
)

target_link_libraries(GreeksEngineTests PRIVATE
    BSMLib
    GTest::gtest_main   
)

include(GoogleTest)
gtest_discover_tests(GreeksEngineTests)