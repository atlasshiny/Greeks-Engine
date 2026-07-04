cmake_minimum_required(VERSION 3.20)
cmake_policy(SET CMP0135 NEW)

project(GreeksEngine)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

FetchContent_MakeAvailable(googletest)

# Main Executable
add_executable(GreeksEngine 
    src/main.cpp
    src/BSMModel.cpp
)
target_include_directories(GreeksEngine PRIVATE include)

# Tests
enable_testing()

add_executable(GreeksEngineTests
    src/BSMModel.cpp # temporary inclusion so tests compile
    tests/bsm_tests.cpp
)

target_link_libraries(GreeksEngineTests PRIVATE
    GTest::gtest_main   
)
target_include_directories(GreeksEngineTests PRIVATE include) # temporary inclusion so tests compile

include(GoogleTest)
gtest_discover_tests(GreeksEngineTests)