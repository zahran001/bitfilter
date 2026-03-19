cmake_minimum_required(VERSION 3.20)
project(bitfilter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler flags ────────────────────────────────────────────────────────────
# -march=native    : enables all CPU features including AVX2, BMI2, POPCNT
# -mavx2 -mpopcnt  : redundant with native, but explicit — documents the hard
#                    dependency; prevents silent fallback if native is ever
#                    swapped out for a cross-compilation baseline
# -fno-omit-frame-pointer : near-zero cost at O3; required for perf call graphs
#                           on Akamai in Week 3 — add it now, never regret it
add_compile_options(
    -O3
    -march=native
    -mavx2
    -mpopcnt
    -fno-omit-frame-pointer
    -Wall
    -Wextra
    -Wpedantic
)

# ── Dependencies via FetchContent ─────────────────────────────────────────────
include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.14.0
)

FetchContent_Declare(
    benchmark
    GIT_REPOSITORY https://github.com/google/benchmark.git
    GIT_TAG        v1.8.4
)

# Don't let benchmark pull in its own copy of gtest and shadow ours
set(BENCHMARK_ENABLE_TESTING     OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(gtest_force_shared_crt       ON  CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(googletest benchmark)

# ── Core library ──────────────────────────────────────────────────────────────
add_library(bitfilter STATIC
    src/segment_store.cpp
    src/query_eval_scalar.cpp
    src/query_eval_avx2.cpp
    src/query_eval_avx2_unroll.cpp
    src/query_eval_avx2_prefetch.cpp
)
target_include_directories(bitfilter PUBLIC include)

# ── Test binary ───────────────────────────────────────────────────────────────
enable_testing()

add_executable(test_correctness
    test/test_correctness.cpp
)
target_link_libraries(test_correctness
    bitfilter
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_correctness)

# ── Benchmark binary ──────────────────────────────────────────────────────────
# benchmark::benchmark_main provides main() — without it the linker fails even
# though the headers compile cleanly. benchmark::benchmark alone is not enough.
add_executable(bench
    bench/bench_main.cpp
)
target_link_libraries(bench
    bitfilter
    benchmark::benchmark
    benchmark::benchmark_main
)
