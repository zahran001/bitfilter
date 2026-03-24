# BitFilter

> A high-performance, SIMD-accelerated audience segmentation engine built in C++20.
> 

> Portfolio project targeting the NVIDIA Developer Technology Engineer (CPU Performance) role.
> 

---

## Problem statement

Modern ad tech platforms (The Trade Desk, Google DV360, LiveRamp) need to answer one question billions of times per day:

> *"Which users belong to Segment A AND Segment B, but NOT Segment C?"*
> 

A naive approach — storing user IDs in lists — requires massive memory and slow pointer chasing. BitFilter solves this with **dense bitmaps**: each segment is a flat array of bits, one bit per user. A boolean query becomes pure bitwise logic across hundreds of megabytes of data, executed at memory-bandwidth speed using SIMD.

---

## Core insight

A standard 64-bit CPU register holds 64 bits. Instead of using those 64 bits to represent *one number*, we use them to represent **64 different users**. With AVX2's 256-bit YMM registers, one `_mm256_and_si256` instruction intersects two segments across **256 users per clock cycle**.

This is **mechanical sympathy** — writing code that respects how the CPU physically moves data from RAM to registers, rather than fighting the hardware.

---

## Hardware target (development machine)

Discovered via Sysinternals `Coreinfo64.exe`:

| Property | Value |
| --- | --- |
| Architecture | x86-64 (Intel 12th/13th Gen hybrid) |
| Logical processors | 12 (2 P-cores w/ HT + 8 E-cores) |
| SIMD support | AVX ✅  AVX2 ✅  BMI2 ✅  AVX-512 ❌ |
| L1 Data cache (P-core) | 48 KB, 12-way assoc, 64B line |
| L1 Data cache (E-core) | 32 KB, 8-way assoc, 64B line |
| L2 (P-core) | 1 MB private per core |
| L2 (E-core) | 2 MB shared per cluster of 4 |
| L3 | 12 MB unified, shared |
| Cache line size | **64 bytes** (all levels) |

**AVX-512 pivot:** Intel disables AVX-512 on hybrid chips because E-cores don't support it — if the OS schedules a thread mid-execution from a P-core to an E-core, the program crashes. The implementation therefore targets **AVX2 (256-bit YMM registers)**. The architecture, alignment strategy, benchmark structure, and blog narrative are fully unchanged.

Secondary target: **ARM SVE** for NVIDIA Grace (ARMv9 SoC), validated via GitHub Actions ARM runners.

---

## Project layout

```
bitfilter/
├── CMakeLists.txt
├── README.md
├── include/
│   ├── aligned_alloc.hpp      ← cross-platform aligned allocation wrapper
│   ├── segment_store.hpp      ← 64B-aligned bitmap storage
│   ├── query_eval.hpp         ← evaluation engine interface
│   └── query_ast.hpp          ← boolean query representation
├── src/
│   ├── segment_store.cpp
│   ├── query_eval_scalar.cpp  ← baseline (no SIMD) — correctness reference
│   ├── query_eval_avx2.cpp    ← primary: 256-bit YMM registers
│   └── query_eval_sve.cpp     ← ARM SVE path for NVIDIA Grace
├── bench/
│   └── bench_main.cpp         ← Google Benchmark harness
└── test/
    └── test_correctness.cpp   ← scalar vs SIMD bit-exact comparison
```

---

## Memory alignment — why it matters

AVX2 aligned loads (`_mm256_load_si256`) require data at **32-byte boundaries**. BitFilter uses **64-byte alignment** throughout to stay cache-line-perfect and future-compatible with AVX-512 targets.

If data is misaligned by even 1 byte, the CPU must fetch two cache lines and stitch them internally — doubling memory traffic and stalling the SIMD unit. Addresses must end in `...00`, `...40`, `...80`, or `...C0` in hex to be 64-byte aligned.

### Cross-platform allocation wrapper

`std::aligned_alloc` is **not reliably supported on Windows/MSVC** — it silently returns `nullptr`, causing an immediate crash on the first `_mm256_load_si256` call. Use this wrapper instead:

```cpp
// include/aligned_alloc.hpp
#pragma once
#include <cstdlib>
#include <memory>
#include <new>

inline void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// RAII wrapper — automatic cleanup, no manual free() needed
struct AlignedDeleter {
    void operator()(void* p) const { aligned_free(p); }
};
using AlignedBuffer = std::unique_ptr<uint64_t[], AlignedDeleter>;

inline AlignedBuffer make_aligned_bitmap(size_t n_words) {
    void* raw = aligned_malloc(n_words * sizeof(uint64_t), 64);
    if (!raw) throw std::bad_alloc{};
    return AlignedBuffer(static_cast<uint64_t*>(raw));
}
```

`SegmentStore` owns an `AlignedBuffer` member — no destructor, no `delete[]`, no platform `#ifdef` in business logic.

---

## Key data structures

```cpp
// include/segment_store.hpp
#pragma once
#include <cstdint>
#include <vector>
#include <string>
#include <unordered_map>
#include "aligned_alloc.hpp"

struct Bitmap {
    AlignedBuffer data;  // 64B-aligned, RAII managed
    size_t        n_words;
    size_t        n_users;

    // Hardware-accelerated popcount using POPCNT instruction (BMI2 confirmed)
    size_t popcount() const noexcept;
};

class SegmentStore {
public:
    explicit SegmentStore(size_t n_users);

    int  add_segment(const std::string& name);
    void set(int seg_id, uint32_t uid);
    void load(int seg_id, const std::vector<uint32_t>& user_ids);

    const Bitmap& get(int seg_id) const;
    int           segment_id(const std::string& name) const;

private:
    size_t n_users_;
    size_t n_words_;
    std::vector<Bitmap>                  bitmaps_;
    std::unordered_map<std::string, int> name_to_id_;
};
```

For 500M users: `500,000,000 / 8 = 62.5 MB` per segment.

To find User #N: `word_index = N / 64`, `bit_index = N % 64`.

---

## Hardware-accelerated popcount

Coreinfo confirmed **BMI2 support**, which guarantees the `POPCNT` hardware instruction. Use `_mm_popcnt_u64` — a single instruction that counts all set bits in a 64-bit word in one cycle, answering "how many users matched?" after every query evaluation:

```cpp
// src/segment_store.cpp
#include <immintrin.h>

size_t Bitmap::popcount() const noexcept {
    size_t count = 0;
    for (size_t i = 0; i < n_words; ++i)
        count += _mm_popcnt_u64(data[i]);
    return count;
}
```

For 500M users (7,812,500 words), this is the difference between ~50ms of naive bit-counting and ~8ms of hardware-accelerated counting. The blog post can show three tiers: naive loop → `__builtin_popcountll` (compiler hint) → `_mm_popcnt_u64` (hardware instruction).

---

## SIMD evaluation — the core

### Scalar baseline (write first, used for correctness validation)

```cpp
// src/query_eval_scalar.cpp
void eval_scalar(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words)
{
    for (size_t i = 0; i < n_words; ++i)
        result[i] = a[i] & b[i] & ~not_c[i];
}
```

### AVX2 implementation (primary path)

```cpp
// src/query_eval_avx2.cpp
#include <immintrin.h>

// Processes 256 users per iteration (4 × uint64_t = 256 bits = 1 YMM register)
void eval_avx2(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words)
{
    size_t i = 0;
    const size_t simd_width = 4; // 4 × uint64_t per YMM register

    for (; i + simd_width <= n_words; i += simd_width) {
        __m256i va  = _mm256_load_si256((__m256i*)(a     + i));
        __m256i vb  = _mm256_load_si256((__m256i*)(b     + i));
        __m256i vc  = _mm256_load_si256((__m256i*)(not_c + i));

        // _mm256_andnot_si256(vc, x) computes (~vc & x) in one instruction
        __m256i tmp = _mm256_and_si256(va, vb);
        __m256i vr  = _mm256_andnot_si256(vc, tmp);

        _mm256_store_si256((__m256i*)(result + i), vr);
    }
    // scalar tail for remainder
    for (; i < n_words; ++i)
        result[i] = a[i] & b[i] & ~not_c[i];
}
```

One loop iteration processes **256 users** in ~1 clock cycle.

---

## Build system

```
cmake_minimum_required(VERSION 3.20)
project(bitfilter CXX)
set(CMAKE_CXX_STANDARD 20)

if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    # -mpopcnt is explicit even though -march=native implies it
    add_compile_options(-mavx2 -mbmi2 -mpopcnt -O3 -march=native)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64")
    add_compile_options(-march=armv9-a+sve2 -O3)
endif()

find_package(benchmark REQUIRED)
find_package(GTest REQUIRED)

add_library(bitfilter
    src/segment_store.cpp
    src/query_eval_scalar.cpp
    src/query_eval_avx2.cpp
    src/query_eval_sve.cpp)

add_executable(bench bench/bench_main.cpp)
target_link_libraries(bench bitfilter benchmark::benchmark)

add_executable(tests test/test_correctness.cpp)
target_link_libraries(tests bitfilter GTest::gtest_main)
```

---

## 4-week execution plan

### Week 1 — Foundation and correctness

- Implement `SegmentStore` using `make_aligned_bitmap()` from `aligned_alloc.hpp`
- Write `eval_scalar` baseline
- Implement synthetic data generator (random bits, configurable density)
- Write correctness tests: scalar vs AVX2 on identical data, assert `memcmp == 0`
- Implement `Bitmap::popcount()` using `_mm_popcnt_u64`
- Set up Google Benchmark harness with `SetBytesProcessed` to track memory bandwidth
- Instrument with `perf stat` to baseline IPC, cache miss rate, memory BW **before** optimizing

### Week 2 — AVX2 SIMD acceleration

- Implement `eval_avx2` with `_mm256_and_si256` + `_mm256_andnot_si256`
- Validate correctness against scalar baseline
- Disable scalar auto-vectorization (`no-tree-vectorize`) to establish true baseline
- Experiment with loop unrolling (`eval_avx2_unroll2`, `eval_avx2_unroll4`) — found slower at DRAM scale due to register pressure
- Add software prefetching: `eval_avx2_prefetch` with `_mm_prefetch((char*)(a + i + 16), _MM_HINT_T0)` — first SIMD variant to beat scalar at DRAM scale (18.2 vs 16.7 GB/s)
- Benchmark popcount tiers: naive loop → `__builtin_popcountll` → `_mm_popcnt_u64` (50× speedup)
- Key finding: workload is memory-bandwidth-bound, not compute-bound. Basic SIMD loses to scalar at 500M; only prefetching wins by hiding DRAM latency

### Week 3 — Threading and bandwidth saturation

- Implement `eval_avx2_prefetch_mt()` using `std::jthread` (coarse-grained workload makes thread pools unnecessary)
- Over-partition bitmap into `4 × n_threads` chunks with `std::atomic<size_t>` chunk index — handles P-core/E-core speed imbalance without detecting core types
- Align all chunk boundaries to 8 words (64 bytes) to prevent false sharing
- MT correctness test: `memcmp` multi-threaded result against single-threaded `eval_avx2_prefetch`
- MT popcount: thread-local `alignas(64)` padded counters, main-thread reduction (no atomics needed)
- Benchmark scaling curve: 1 / 2 / 4 / 8 / 12 threads at 500M users
- Core-type experiment: P-cores only (2T) vs P-cores + HT (4T) vs all cores (12T)
- Measure achieved GB/s vs theoretical DRAM bandwidth ceiling — goal is to show saturation
- NUMA deferred: single-socket dev machine, not measurable. Add on Akamai multi-socket if available

### Week 4 — Benchmarking and write-up

- Benchmark against CRoaring (industry-standard Roaring Bitmap library)
- Benchmark against DuckDB and SQLite for equivalent queries
- Build roofline model: plot achieved GB/s vs theoretical memory bandwidth ceiling
- Write developer blog post (problem → measurement → optimization → result)
- Publish GitHub repo with CI matrix: `ubuntu-latest` (x86 AVX2) + ARM runner (SVE)

---

## Benchmark design

```cpp
// bench/bench_main.cpp
#include <benchmark/benchmark.h>
#include "segment_store.hpp"
#include "query_eval.hpp"

static constexpr size_t N_USERS = 500'000'000ULL;

static void BM_Scalar(benchmark::State& state) {
    for (auto _ : state) {
        eval_scalar(a, b, c, result, n_words);
        benchmark::ClobberMemory();
    }
    // 3 bitmaps read + 1 written = 4 × 62.5 MB per iteration
    state.SetBytesProcessed(state.iterations() * 4 * (N_USERS / 8));
}

BENCHMARK(BM_Scalar)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AVX2)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AVX2_Prefetch)->Unit(benchmark::kMillisecond);
BENCHMARK(BM_AVX2_Parallel)->Threads(12)->Unit(benchmark::kMillisecond);
```

Measured results for a 3-segment query over 500M users (Week 2 actuals, then Week 3 target):

| Implementation | Latency | Throughput | Status |
| --- | --- | --- | --- |
| Scalar (de-vectorized) | 13.9 ms | 16.7 GB/s | ✅ Measured |
| AVX2 (1 register) | 18.6 ms | 12.6 GB/s | ✅ Measured — loses to scalar (DRAM-bound) |
| AVX2 + prefetch | 12.8 ms | 18.2 GB/s | ✅ Measured — 1.09× over scalar |
| AVX2 + prefetch + 12 threads | TBD | Target: near DRAM ceiling | ⬜ Week 3 |

---

## Comparison targets (Week 4)

| System | Type | Expected result |
| --- | --- | --- |
| Scalar baseline | Dense bitmap, no SIMD | Our floor |
| AVX2 BitFilter | Dense bitmap + SIMD | Our ceiling |
| CRoaring | Sparse bitmap (industry std) | Better at low-density segments |
| DuckDB | Columnar SQL engine | Higher-level abstraction overhead |
| SQLite | Row-store SQL | Worst case — demonstrates why bitmaps matter |

---

## Mapping to NVIDIA

| Requirement | How BitFilter covers it |
| --- | --- |
| CPU architecture (x86, ARM) | Dual path: AVX2 (x86) + ARM SVE (Grace) |
| Memory subsystem (cache, DRAM) | 64B alignment, cache-line analysis, prefetch tuning |
| SIMD / CPU intrinsics | `_mm256_and_si256`, `_mm256_andnot_si256`, `_mm_popcnt_u64`, `_mm_prefetch` |
| Low-level parallel programming | Over-partitioned chunk dispatch, cache-line-aligned boundaries, core-type scaling analysis |
| Database / data-intensive workloads | Models ad audience segmentation engines directly |
| Publish optimization techniques | Developer blog post + open-source GitHub repo |
| Influence hardware design | Roofline model quantifies where future CPU improvements matter |

---

---

## Development environment

| Concern | Decision |
| --- | --- |
| Daily development | WSL2 Ubuntu 24.04 on HP laptop |
| Native Windows | ❌ Not used — MSVC intrinsic friction, no perf, no numactl |
| CI | GitHub Actions `ubuntu-latest` (x86 AVX2) + ARM runner (SVE) |
| Week 3 perf profiling | ✅ Akamai Cloud $100 free credit — bare metal Linux, full hardware counters |

### WSL2 environment — confirmed working

```
Ubuntu 24.04.4 LTS (Noble)
Kernel: 5.15.146.1-microsoft-standard-WSL2
g++ 13.3.0
cmake 3.28.3
AVX2 ✅  POPCNT ✅  Google Benchmark 1.8.3 ✅
```

### Google Benchmark link flags (critical)

The `-lbenchmark_main` flag is required when using `BENCHMARK_MAIN()`. Missing it causes a linker failure even though headers are found. Always use:

```
g++ ... -lbenchmark -lbenchmark_main -lpthread
```

In CMake this is handled automatically via `target_link_libraries(bench bitfilter benchmark::benchmark)`.

### perf hardware counters on WSL2

`perf` hardware counters (`cycles`, `cache-misses`, `instructions`, IPC) are **blocked on WSL2's Microsoft kernel** — no fix exists for Intel 12th/13th Gen hybrid chips as of March 2026 (confirmed via GitHub issue #12836). This affects Week 3 only.

**✅ DECISION LOCKED: Use Akamai Cloud for Week 3 perf profiling.**

- Sign up at https://www.akamai.com/lp/free-credit-100-5000 (GitHub SSO, $100 free credit, 60 days)
- Spin up Ubuntu 24.04 shared CPU instance (~$0.018/hr)
- Clone repo, run `sudo perf stat -e cycles,instructions,cache-misses ./bench`, destroy instance
- Total cost: < $0.10 of the $100 credit

Weeks 1, 2, and 4 are fully unaffected — WSL2 is sufficient for correctness, SIMD, and benchmark throughput numbers.

---

## References

- [Intel Intrinsics Guide](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
- [CRoaring — Roaring Bitmaps in C](https://github.com/RoaringBitmap/CRoaring)
- [Google Benchmark](https://github.com/google/benchmark)
- [NVIDIA Grace CPU architecture](https://www.nvidia.com/en-us/data-center/grace-cpu)
- [Agner Fog optimization manuals](https://www.agner.org/optimize)
- [Akamai Cloud free credit](https://www.akamai.com/lp/free-credit-100-5000)