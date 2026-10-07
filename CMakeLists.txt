cmake_minimum_required(VERSION 3.20)
project(cardinality_sketches CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(benchmark QUIET)
if(NOT benchmark_FOUND)
    message(STATUS "benchmark not found on system — fetching via FetchContent")
    include(FetchContent)
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG        v1.9.5
    )
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(benchmark)
else()
    message(STATUS "Using system benchmark: ${benchmark_DIR}")
endif()

add_library(sketches cpp/src/dummy_sketch.cpp)
target_include_directories(sketches
    PUBLIC  cpp/include
    PRIVATE cpp/external
)

add_executable(benchmarks cpp/bench/bench.cpp)
target_include_directories(benchmarks PRIVATE cpp/external)
target_link_libraries(benchmarks PRIVATE sketches benchmark::benchmark)

enable_testing()
add_executable(test_dummy_sketch cpp/tests/test_dummy_sketch.cpp)
target_include_directories(test_dummy_sketch PRIVATE cpp/external)
target_link_libraries(test_dummy_sketch PRIVATE sketches)
add_test(NAME dummy_sketch COMMAND test_dummy_sketch)
