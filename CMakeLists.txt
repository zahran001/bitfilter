cmake_minimum_required(VERSION 3.20)
project(bitfilter LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ── Compiler flags ────────────────────────────────────────────────────────────
# Common flags for all architectures
add_compile_options(
    -O3
    -fno-omit-frame-pointer
    -Wall
    -Wextra
    -Wpedantic
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    # -march=native    : enables all CPU features including AVX2, BMI2, POPCNT
    # -mavx2 -mpopcnt  : redundant with native, but explicit — documents the hard
    #                    dependency; prevents silent fallback if native is ever
    #                    swapped out for a cross-compilation baseline
    add_compile_options(-march=native -mavx2 -mpopcnt)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    # ARM SVE — Ampere Altra (GitHub ARM runners) supports ARMv8 + SVE
    add_compile_options(-march=armv8-a+sve -O3)
endif()

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

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    FetchContent_Declare(
        croaring
        GIT_REPOSITORY https://github.com/RoaringBitmap/CRoaring.git
        GIT_TAG        v4.6.1
    )
    # CRoaring: disable extras we don't need
    set(ROARING_BUILD_LTO            OFF CACHE BOOL "" FORCE)
    set(ENABLE_ROARING_TESTS         OFF CACHE BOOL "" FORCE)

    FetchContent_MakeAvailable(googletest benchmark croaring)
else()
    FetchContent_MakeAvailable(googletest benchmark)
endif()

# ── Core library ──────────────────────────────────────────────────────────────
set(BITFILTER_SOURCES
    src/segment_store.cpp
    src/query_eval_scalar.cpp
)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
    list(APPEND BITFILTER_SOURCES
        src/query_eval_avx2.cpp
        src/query_eval_avx2_unroll.cpp
        src/query_eval_avx2_prefetch.cpp
        src/query_eval_mt.cpp
    )
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|ARM64")
    list(APPEND BITFILTER_SOURCES
        src/query_eval_sve.cpp
    )
endif()

add_library(bitfilter STATIC ${BITFILTER_SOURCES})
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
gtest_discover_tests(test_correctness DISCOVERY_TIMEOUT 120)

# ── Benchmark binaries (x86 only — use AVX2 intrinsics) ──────────────────────
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
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

    add_executable(bench_popcount
        bench/bench_popcount.cpp
    )
    target_link_libraries(bench_popcount
        bitfilter
        benchmark::benchmark
        benchmark::benchmark_main
    )

    # ── DuckDB comparative benchmark ───────────────────────────────────────
    # Uses prebuilt libduckdb.so (v1.5.1) — the 25 MB amalgamation source OOMs
    # during compilation on machines with <12 GB RAM.
    # One-liner to fetch:
    #   mkdir -p bench/duckdb && curl -sL https://github.com/duckdb/duckdb/releases/download/v1.5.1/libduckdb-linux-amd64.zip | python3 -c "import sys,zipfile,io; z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())); [z.extract(f,'bench/duckdb/') for f in ('duckdb.hpp','libduckdb.so')]"
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/bench/duckdb/libduckdb.so")
        # Import prebuilt shared library
        add_library(duckdb SHARED IMPORTED)
        set_target_properties(duckdb PROPERTIES
            IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/bench/duckdb/libduckdb.so"
            INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/bench/duckdb"
        )

        add_executable(bench_duckdb bench/bench_duckdb.cpp)
        target_include_directories(bench_duckdb PRIVATE bench/duckdb)
        target_link_libraries(bench_duckdb
            bitfilter
            duckdb
            benchmark::benchmark
            benchmark::benchmark_main
        )
    endif()

    # ── CRoaring comparative benchmark ──────────────────────────────────────
    add_executable(bench_croaring
        bench/bench_croaring.cpp
    )
    target_link_libraries(bench_croaring
        bitfilter
        roaring
        benchmark::benchmark
        benchmark::benchmark_main
    )

    # ── SQLite comparative benchmark ───────────────────────────────────────
    # Requires libsqlite3-dev: sudo apt-get install -y libsqlite3-dev
    find_package(SQLite3)
    if(SQLite3_FOUND)
        add_executable(bench_sqlite bench/bench_sqlite.cpp)
        target_link_libraries(bench_sqlite
            bitfilter
            SQLite::SQLite3
            benchmark::benchmark
            benchmark::benchmark_main
        )
    endif()
endif() # x86_64
