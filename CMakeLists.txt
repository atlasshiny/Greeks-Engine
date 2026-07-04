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
# enable_testing()

# add_executable(GreeksEngineTests
#     tests/bsm_test.cpp
# )

# target_link_libraries(GreeksEngineTests PRIVATE
#     GTest::gtest_main  
#     Threads::Threads   
# )

# include(GoogleTest)
# gtest_discover_tests(OrderBookTests)