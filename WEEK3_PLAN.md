# Week 3 — Threading and bandwidth saturation

## Goal

Saturate DRAM bandwidth by parallelizing `eval_avx2_prefetch` across all 12 logical cores.

Single-threaded AVX2+prefetch delivers 18.2 GB/s. The theoretical DRAM ceiling on this
machine (DDR4-2660 dual-channel, confirmed via CPU-Z) is **42.6 GB/s peak**.
Practical achievable with mixed read/write traffic: **~30-35 GB/s**.
Threading is the only way to close that gap — the compute is trivial, the memory bus
is the bottleneck.

**Hardware memory specs (CPU-Z verified):**
- DRAM frequency: 1330.1 MHz (DDR4-2660 effective, 2 × 1330 MT/s)
- Channels: 2 × 64-bit (dual channel)
- Per-channel bandwidth: 2660 MT/s × 8 bytes = 21.3 GB/s
- Theoretical peak: 2 × 21.3 = **42.6 GB/s**
- CAS latency: CL19, tRCD 19, tRP 19, tRAS 43 (19-19-19-43 @ 2T command rate)

**Success metric:** achieve >80% of practical DRAM bandwidth (~28+ GB/s) at 12 threads.

---

## Where we are (inputs from Week 2)

| Item | Value |
|------|-------|
| Best single-threaded variant | `eval_avx2_prefetch` — 12.8 ms, 18.2 GB/s at 500M |
| Scalar baseline | 13.9 ms, 16.7 GB/s at 500M |
| CPU topology | 2 P-cores (HT) + 8 E-cores = 12 logical processors |
| Alignment invariant | All bitmaps 64-byte aligned via `make_aligned_bitmap()` |
| Existing function signature | `eval_avx2_prefetch(a, b, not_c, result, n_words)` |
| Existing tests | 16 passing (4 unit + 12 parametrized correctness) |
| Existing benchmarks | 5 variants × 3 scales + 3 popcount tiers |

---

## Chunk breakdown

### Chunk 1 — `eval_avx2_prefetch_mt()` core implementation

**New file:** `src/query_eval_mt.cpp`
**New header additions:** `include/query_eval.hpp`

```cpp
// Signature
void eval_avx2_prefetch_mt(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words,
    unsigned n_threads);
```

**Design decisions:**

1. **Threading model: `std::jthread`** — spawn per call, join on scope exit.
   At 500M users each thread runs for ~1ms+. Thread creation (~50μs) is <1% overhead.
   Thread pools solve latency problems; this is a throughput/bandwidth problem.

2. **Over-partitioning with atomic chunk index** — divide bitmap into `4 × n_threads`
   chunks. Each thread loops on `next_chunk.fetch_add(1, std::memory_order_relaxed)`.
   This automatically balances P-cores (fast) vs E-cores (slow) without detecting
   core types. A P-core simply grabs more chunks before all chunks are claimed.

3. **Cache-line-aligned chunk boundaries** — each chunk starts at a multiple of
   8 words (64 bytes). Prevents false sharing where two threads write adjacent
   words in the same cache line. Last chunk handles any remainder.

4. **Each chunk calls `eval_avx2_prefetch`** on its sub-range. No new SIMD code —
   reuse the proven single-threaded kernel. The MT wrapper is purely a dispatch layer.

**Pseudocode:**

```cpp
void eval_avx2_prefetch_mt(..., unsigned n_threads) {
    constexpr size_t CL_WORDS = 8;  // 64 bytes / 8 bytes per word
    const unsigned n_chunks = 4 * n_threads;
    const size_t chunk_words = (n_words / n_chunks / CL_WORDS) * CL_WORDS;

    std::atomic<unsigned> next_chunk{0};

    auto worker = [&]() {
        while (true) {
            unsigned c = next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (c >= n_chunks) break;

            size_t begin = c * chunk_words;
            size_t end   = (c == n_chunks - 1) ? n_words : begin + chunk_words;
            size_t count = end - begin;

            eval_avx2_prefetch(a + begin, b + begin, not_c + begin,
                               result + begin, count);
        }
    };

    std::vector<std::jthread> threads;
    threads.reserve(n_threads - 1);
    for (unsigned t = 1; t < n_threads; ++t)
        threads.emplace_back(worker);
    worker();  // main thread participates
    // jthread destructor joins automatically
}
```

**Edge cases to handle:**
- `n_threads == 0` or `n_threads == 1` → fall through to single-threaded `eval_avx2_prefetch`
- `n_words < n_chunks * CL_WORDS` → reduce n_chunks to avoid empty chunks
- Remainder words in last chunk (not a multiple of 4) → handled by `eval_avx2_prefetch`'s scalar tail

**CMake:** add `src/query_eval_mt.cpp` to `libbitfilter` sources. Link `-lpthread` (already
implied by `std::jthread` on Linux, but explicit doesn't hurt).

**Build and verify it compiles before moving on.**

---

### Chunk 2 — MT correctness test

**File:** `test/test_correctness.cpp` (add new test, don't create a new file)

```cpp
// Add eval_avx2_prefetch_mt to the parametrized correctness suite.
// Key assertion: memcmp(result_single_threaded, result_multi_threaded) == 0
```

**Test cases (parametrized):**
- Thread counts: 1, 2, 4, 12
- Word counts: 8192 (aligned), 8195 (remainder), 1 (degenerate)
- Cross-product: 4 × 3 = 12 new test cases

**Why test n_threads=1?** Verifies the MT wrapper's fallback path is correct.

**Why test n_words=1?** With 48 chunks and 1 word, most chunks are empty. Tests
the "more chunks than words" edge case.

**Run all tests, confirm 28 passing (16 existing + 12 new).**

---

### Chunk 3 — MT benchmark (scaling curve)

**File:** `bench/bench_main.cpp` (add MT benchmarks to existing file)

Add `Avx2PrefetchMT` fixture benchmark, parametrized by `{n_users, n_threads}`:

| n_users | n_threads values |
|---------|-----------------|
| 500M | 1, 2, 4, 8, 12 |

**Why only 500M?** Threading is irrelevant at L3-resident scale (1M). The overhead
of spawning threads would dominate. The interesting story is DRAM-bound scaling.

**Metrics to capture:**
- Wall time (ms) per thread count
- GB/s throughput (same `SetBytesProcessed` formula: 4 × n_words × 8)
- Derive scaling efficiency: `speedup(N) = time(1) / time(N)`, `efficiency = speedup / N`

**Run the benchmark, record results.**

---

### Chunk 4 — MT popcount with thread-local reduction

**New function in `src/query_eval_mt.cpp`:**

```cpp
uint64_t popcount_mt(const uint64_t* bitmap, size_t n_words, unsigned n_threads);
```

**Design:**
- Same over-partitioning dispatch as eval
- Each thread accumulates a local popcount for its chunks
- Thread-local counters stored in a `alignas(64)` padded array (prevents false sharing):

```cpp
struct alignas(64) PaddedCounter { uint64_t value = 0; };
```

- Main thread sums the array after join — no atomics needed
- Each thread assigned a fixed slot index (not shared with chunk dispatch — the
  slot is per-thread, the chunks are dynamically dispatched)

**Correctness test:** compare `popcount_mt(result, n_words, 12)` against
`popcount(result, n_words)` — must be exactly equal.

**Benchmark:** add to `bench_popcount.cpp` — compare ST vs 12-thread popcount at 500M.

---

### Chunk 5 — Core-type experiment and analysis

**Experiments to run (all at 500M users):**

| Config | Threads | What it tests |
|--------|---------|---------------|
| P-cores only | 2 | Single P-core pair, no contention |
| P-cores + HT | 4 | Does hyperthreading help on memory-bound work? |
| All cores | 12 | Full hardware utilization |

**How to pin threads:** `pthread_setaffinity_np` or `sched_setaffinity` to restrict
which logical cores are used. Core IDs for P-cores vs E-cores can be read from
`/sys/devices/system/cpu/cpu*/topology/core_type` (or `intel_pstate` sysfs).

**If pinning is too complex on WSL2:** just benchmark 2/4/12 threads without pinning
and note that the OS scheduler distributes across core types. The scaling curve still
tells the story even without pinning.

**Theoretical bandwidth reference (CPU-Z verified, for comparison with measured results):**

```
RAM:        DDR4-2660, dual-channel, CL19-19-19-43 @ 2T
Peak BW:    2660 MT/s × 16 bytes = 42.6 GB/s

Eval workload (A & B & ~C):
  3 reads + 1 write = 4 cache lines per 64 bytes of progress
  500M users = 62.5 MB per bitmap = 250 MB total bus traffic
  Theoretical floor: 250 MB / 42.6 GB/s = ~5.9 ms

Single-threaded measured: 12.8 ms → ~46% of peak
80% of peak would be:     ~7.3 ms
Theoretical floor:        ~5.9 ms (assumes 100% bus utilization, no overhead)
```

These are back-of-envelope numbers. Actual results will differ due to refresh
cycles, bus turnaround, TLB misses, and hybrid core scheduling. Compare measured
results against this reference to see how close we get.

**Deliverables:**
- Scaling curve table in PROGRESS.md
- Bandwidth utilization percentage
- Core-type breakdown (if pinning works)
- Update plan.md target table with actual Week 3 numbers

---

## Files touched per chunk

| Chunk | New/Modified files |
|-------|--------------------|
| 1 | `src/query_eval_mt.cpp` (new), `include/query_eval.hpp` (add decl), `CMakeLists.txt` (add source) |
| 2 | `test/test_correctness.cpp` (add MT tests) |
| 3 | `bench/bench_main.cpp` (add MT benchmarks) |
| 4 | `src/query_eval_mt.cpp` (add popcount_mt), `include/query_eval.hpp` (add decl), `test/test_correctness.cpp` (add test), `bench/bench_popcount.cpp` (add benchmark) |
| 5 | `PROGRESS.md` (results), `plan.md` (update targets) |

---

## Risk register

| Risk | Likelihood | Mitigation |
|------|-----------|------------|
| WSL2 thread scheduling is unpredictable (P/E mapping) | Medium | Over-partitioning handles imbalance; document if pinning fails |
| Scaling plateaus early (e.g., 4 threads saturates bus) | Medium | That's actually a good result — proves memory-bound ceiling |
| Thread creation overhead visible at small scales | Low | Only benchmark MT at 500M; note in docs that MT is for DRAM-bound workloads |
| `jthread` not available in g++ 13.3 | Very low | g++ 13 fully supports C++20 jthread; verified by `-std=c++20` flag |

---

## Definition of done

- [ ] `eval_avx2_prefetch_mt()` implemented with over-partitioned chunk dispatch
- [ ] 28+ tests passing (16 existing + 12 MT correctness)
- [ ] Scaling curve benchmarked: 1/2/4/8/12 threads at 500M
- [ ] `popcount_mt()` implemented and tested
- [ ] Achieved GB/s vs theoretical DRAM ceiling reported
- [ ] PROGRESS.md updated with Week 3 results
- [ ] plan.md target table updated with actual numbers
