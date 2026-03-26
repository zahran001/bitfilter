# BitFilter — Progress Tracker

---

## Week 1 — Foundation and correctness ✅

| Chunk | Deliverable | Status |
|-------|-------------|--------|
| 1 | Project skeleton, CMake, directory structure, stub files | ✅ Done |
| 2 | SegmentStore, aligned_alloc, set_bit/get_bit, popcount, DataGen | ✅ Done |
| 3 | eval_scalar + data generator (already delivered in Chunks 1-2) | ✅ Done |
| 4 | eval_avx2 correctness tests — bit-exact memcmp at 3 word counts | ✅ Done |
| 5 | Google Benchmark harness — scalar vs AVX2 at 1M/10M/500M users | ✅ Done |

**Week 1 baseline (Intel 12th Gen, WSL2):**

| Benchmark | 1M users | 10M users | 500M users |
|-----------|----------|-----------|------------|
| Scalar | 0.007 ms (65.4 GB/s) | 0.154 ms (30.2 GB/s) | 31.1 ms (7.5 GB/s) |
| AVX2 | 0.007 ms (65.6 GB/s) | 0.211 ms (22.0 GB/s) | 25.7 ms (9.1 GB/s) |

AVX2 only 21% faster than scalar at 500M — likely because `-O3 -march=native` auto-vectorizes the scalar path.

> **Post-hoc verification (2026-03-22):** The 7.5 GB/s scalar number was a
> cold-CPU/thermal-throttle artifact, not a real measurement. Re-running the
> auto-vectorized scalar (attribute removed) yields **18.0 GB/s median** at 500M —
> matching the de-vectorized baseline. Assembly inspection confirmed GCC does
> emit AVX2 (`ymm`/`vpand`) without `no-tree-vectorize`, but at DRAM scale the
> CPU is memory-bound either way, so the instruction mix doesn't affect throughput.
> See `docs/VERIFICATION.md` for the full procedure and decision matrix.

---

## Week 2 — AVX2 optimization and measurement

| Chunk | Deliverable | Status |
|-------|-------------|--------|
| 1 | Disable auto-vectorization of eval_scalar, re-baseline | ✅ Done |
| 2 | Loop unrolling — eval_avx2_unroll2 + eval_avx2_unroll4, correctness tests | ✅ Done |
| 3 | Benchmark unrolled variants at all three sizes | ✅ Done |
| 4 | Software prefetching — eval_avx2_prefetch, correctness test + benchmark | ✅ Done |
| 5 | Popcount benchmark tiers — naive vs __builtin vs _mm_popcnt_u64 | ✅ Done |

**Chunk 1 — Disable scalar auto-vectorization + re-baseline** ✅

Added `__attribute__((optimize("no-tree-vectorize")))` to `eval_scalar`. Re-baselined:

| Benchmark | 1M users | 10M users | 500M users |
|-----------|----------|-----------|------------|
| Scalar (de-vectorized) | 0.010 ms (47.1 GB/s) | 0.133 ms (35.2 GB/s) | 14.4 ms (16.2 GB/s) |
| AVX2 | 0.005 ms (101.4 GB/s) | 0.110 ms (42.3 GB/s) | 18.8 ms (12.4 GB/s) |

At 1M (L3-resident): AVX2 is **2x faster** — compute-bound, SIMD wins clearly.
At 500M (DRAM-bound): scalar is actually faster (16.2 vs 12.4 GB/s). The single-register
AVX2 loop has more instruction overhead per word than the compiler's optimized scalar loop.
This is the gap that unrolling and prefetching aim to close in Chunks 2-4.

**Chunks 2+3 — Loop unrolling + benchmarks** ✅

Added `eval_avx2_unroll2` (2 YMM/iter) and `eval_avx2_unroll4` (4 YMM/iter) in
`src/query_eval_avx2_unroll.cpp`. Correctness validated via memcmp (9 tests, all passing).

Benchmark results (median CPU time, 3 reps):

| Variant | 1M (L3) | 500M (DRAM) |
|---------|---------|-------------|
| Scalar | 0.009 ms (49.6 GB/s) | 13.6 ms (17.1 GB/s) |
| AVX2 1x | 0.005 ms (100.4 GB/s) | 18.5 ms (12.6 GB/s) |
| AVX2 Unroll2 | 0.005 ms (97.4 GB/s) | 22.6 ms (10.3 GB/s) |
| AVX2 Unroll4 | 0.006 ms (72.3 GB/s) | 25.1 ms (9.3 GB/s) |

**Finding:** Unrolling makes things *worse* at DRAM scale. More unrolling = more register
pressure and instruction overhead, but the CPU is already waiting on DRAM. The original
1-register AVX2 loop remains the best SIMD variant. At L3 scale, AVX2 1x wins at 100 GB/s.
The scalar path wins at 500M because GCC's scalar codegen has less overhead per word.

**Chunk 4 — Software prefetching** ✅

Added `eval_avx2_prefetch` in `src/query_eval_avx2_prefetch.cpp`. Based on the 1-register
AVX2 loop (best SIMD variant) with `_mm_prefetch(_MM_HINT_T0)` 16 words (128 bytes) ahead.
16 correctness tests passing.

| Variant | 1M (L3) | 500M (DRAM) |
|---------|---------|-------------|
| Scalar | 0.010 ms (46.7 GB/s) | 13.9 ms (16.7 GB/s) |
| AVX2 | 0.007 ms (62.3 GB/s) | 18.6 ms (12.6 GB/s) |
| **AVX2 Prefetch** | **0.005 ms (98.6 GB/s)** | **12.8 ms (18.2 GB/s)** |

**Finding:** Prefetch is the first SIMD variant to beat scalar at DRAM scale (18.2 vs 16.7 GB/s).
At L3, it matches the best AVX2 result with the lowest variance (1.3% CV). This is the
production-quality eval path.

**Chunk 5 — Popcount tiers for blog narrative** ✅

Separate `bench_popcount` binary with three implementations at 500M users:

| Tier | Implementation | Time | Throughput | Speedup |
|------|---------------|------|------------|---------|
| 1 | Naive (shift+mask loop) | 212 ms | 281 MB/s | baseline |
| 2 | `__builtin_popcountll` | 4.31 ms | 13.5 GB/s | **49x** |
| 3 | `_mm_popcnt_u64` | 4.23 ms | 13.8 GB/s | **50x** |

Builtin and hardware are nearly identical — GCC with `-mpopcnt` lowers `__builtin_popcountll`
to the same POPCNT instruction. The 50x gap from naive is the blog headline number.

---

## Week 3 — Threading and bandwidth saturation

| Chunk | Deliverable | Status |
|-------|-------------|--------|
| 1 | `eval_avx2_prefetch_mt()` — over-partitioned chunk dispatch with `std::jthread` | ✅ Done |
| 2 | MT correctness tests — 12 eval + 12 popcount (40 total tests passing) | ✅ Done |
| 3 | MT eval benchmark — scaling curve at 1/2/4/8/12 threads, 500M users | ✅ Done |
| 4 | `popcount_mt()` with thread-local padded counters + benchmark | ✅ Done |
| 5 | Analysis and results documentation | ✅ Done |

**Hardware reference (CPU-Z verified):**
- DDR4-2660, dual-channel, CL19-19-19-43 @ 2T
- Theoretical peak: 42.6 GB/s
- Practical ceiling (mixed read/write): ~30-35 GB/s

**Chunk 1 — `eval_avx2_prefetch_mt()` implementation** ✅

New `src/query_eval_mt.cpp` with over-partitioned chunk dispatch:
- `4 × n_threads` chunks, cache-line aligned (8 words = 64 bytes)
- Dynamic load balancing via `std::atomic<unsigned> fetch_add(relaxed)`
- P-cores grab more chunks than E-cores automatically
- `std::jthread` (spawn per call, join on scope exit)
- Main thread participates as worker
- Fallback to single-threaded for `n_threads <= 1` or degenerate sizes

**Chunk 2 — MT correctness tests** ✅

24 new parametrized tests (12 eval + 12 popcount):
- Thread counts: {1, 2, 4, 12} × word counts: {8192, 8195, 1}
- Eval: `memcmp(result_st, result_mt) == 0`
- Popcount: `popcount_mt() == popcount()`
- All 40 tests passing.

**Chunks 3+4 — MT benchmarks and results** ✅

> **Note:** Initial benchmark numbers were corrected after verification (see
> `WEEK3_VERIFICATION.md`). The original session overstated eval scaling due to
> a cold-baseline artifact. All numbers below are from the verified back-to-back run.

Eval scaling at 500M users (verified, back-to-back single invocation):

| Threads | Wall time | Throughput (GiB/s) | Speedup vs ST |
|---------|-----------|-------------------|---------------|
| ST (standalone) | 10.7 ms | 22.7 GiB/s | 1.00x |
| 1 (MT fallback) | 10.5 ms | 23.1 GiB/s | 1.02x |
| 2 | 10.9 ms | 23.1 GiB/s | 0.98x |
| 4 | 15.8 ms | 16.4 GiB/s | 0.68x |
| 8 | 11.4 ms | 22.6 GiB/s | 0.94x |
| 12 | 12.1 ms | 20.9 GiB/s | 0.88x |

**Analysis: Threading provides zero benefit for eval.** Single-threaded AVX2+prefetch
already saturates the memory bus. Accounting for write-allocate (x86 stores to cold
cache lines trigger a read-for-ownership), actual DRAM traffic at 1T is:
- Application: 4 × 62.5 MB = 250 MB → 23.4 GB/s reported
- Actual DRAM: 5 × 62.5 MB = 312.5 MB → **~29 GB/s consumed** (68% of 42.6 GB/s peak)

This is already at the practical ceiling (~30-35 GB/s). Adding threads only adds overhead.
The 4T regression to 15.8 ms is likely P-core hyperthread contention.

Popcount MT scaling at 500M users (verified):

| Threads | Wall time | Throughput (GiB/s) | Speedup vs ST |
|---------|-----------|-------------------|---------------|
| ST (standalone) | 3.80 ms | 15.9 GiB/s | 1.00x |
| 1 (MT fallback) | 4.08 ms | 14.8 GiB/s | 0.93x |
| 2 | 2.26 ms | 27.1 GiB/s | **1.68x** |
| 4 | 2.32 ms | 28.4 GiB/s | **1.64x** |
| 8 | 2.23 ms | 29.2 GiB/s | **1.70x** |
| 12 | 2.40 ms | 28.7 GiB/s | **1.58x** |

**Analysis: Popcount scaling is real.** Peak 1.70x at 8 threads. Unlike eval, single-
threaded popcount only consumes 17.1 GB/s (40% of peak) — there is headroom for
additional threads. No write-allocate (read-only), no bus turnaround penalty.

**Chunk distribution (V3 diagnostic):**

Over-partitioned dispatch was verified via `diag_chunks.cpp`:
- No-work mode: 100% imbalance (one thread claims all chunks before others start)
- With-work mode at 500M: 0% imbalance at 2T (perfect 4/4), 71% at 12T (2-7 range)
- `std::jthread` spawn stagger (~50-100μs per thread) causes early threads to grab
  disproportionate chunks. This explains the 12T regression.

**Core-type experiment (Chunk 5):**

Thread pinning (`pthread_setaffinity_np`) was not attempted. The 2/4/8/12 scaling curve
already implicitly covers P-core and E-core participation — the OS scheduler distributes
threads across both core types. Since the workload is memory-bandwidth-bound (not
compute-bound), core type has minimal impact: both P-cores and E-cores can saturate
their share of the memory bus. The bandwidth saturation story is clear without pinning.

**1T MT fallback overhead (popcount):**

The MT wrapper at 1T reported 4.08 ms vs standalone 3.80 ms (7% overhead). This is
benchmark fixture variance (different SetUp/TearDown paths, different iteration counts
165 vs 176), not real overhead — the code path hits `return popcount(bitmap, n_words)`
directly with no allocation or atomic setup. No optimization needed.

**Key takeaways:**
1. **Eval is already bus-saturated at 1T** — write-allocate traffic puts actual DRAM
   consumption at ~29 GB/s, near the practical ceiling. Threading cannot help.
2. **Popcount scales because 1T leaves headroom** — read-only, 17 GB/s at 1T, peaks
   at 31 GB/s with 8 threads (73% of theoretical).
3. **Over-partitioned dispatch works at 2T, degrades at 12T** — thread spawn stagger
   creates chunk imbalance at high thread counts.
4. **Initial benchmarks were misleading** — rigorous verification caught a cold-baseline
   artifact that inflated apparent eval scaling from 1.22x to the true 0.98x.
5. **The negative result is the finding** — proving bandwidth saturation at 1T and
   explaining *why* parallelism doesn't help is more valuable than a fake speedup.

---

## Week 4 — Benchmarking and write-up

| Chunk | Deliverable | Status |
|-------|-------------|--------|
| TBD | CRoaring/DuckDB comparison, roofline model, blog post, CI matrix | ⬜ |
