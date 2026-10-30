cmake_minimum_required(VERSION 3.20)
project(cardinality_sketches CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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

# xxHash has no CMakeLists.txt at its root, so it is populated directly.
cmake_policy(SET CMP0169 OLD)
include(FetchContent)
FetchContent_Declare(
    xxhash
    GIT_REPOSITORY https://github.com/Cyan4973/xxHash.git
    GIT_TAG        v0.8.4
    GIT_SHALLOW    TRUE
)
FetchContent_GetProperties(xxhash)
if(NOT xxhash_POPULATED)
    FetchContent_Populate(xxhash)
endif()

add_library(sketches
    cpp/src/dummy_sketch.cpp cpp/src/linear_counting.cpp
    cpp/src/dummy_sketch.cpp cpp/src/kmv.cpp)
target_include_directories(sketches
    PUBLIC  cpp/include ${xxhash_SOURCE_DIR}
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

add_executable(test_hash cpp/tests/test_hash.cpp)
target_link_libraries(test_hash PRIVATE sketches)
add_test(NAME hash COMMAND test_hash)

add_executable(test_linear_counting cpp/tests/test_linear_counting.cpp)
target_link_libraries(test_linear_counting PRIVATE sketches)
add_test(NAME linear_counting COMMAND test_linear_counting)

find_program(CCMAKE_EXECUTABLE NAMES ccmake)
if(CCMAKE_EXECUTABLE)
    add_custom_target(ccmake
        COMMAND ${CCMAKE_EXECUTABLE} ${CMAKE_SOURCE_DIR}
        WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
        USES_TERMINAL
        COMMENT "Launching ccmake in ${CMAKE_BINARY_DIR}"
    )
    message(STATUS "ccmake found: ${CCMAKE_EXECUTABLE} — target 'ccmake' available")
else()
    message(STATUS "ccmake not found — skipping 'ccmake' target")
endif()
