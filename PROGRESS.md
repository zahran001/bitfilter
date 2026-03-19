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

---

## Week 2 — AVX2 optimization and measurement

| Chunk | Deliverable | Status |
|-------|-------------|--------|
| 1 | Disable auto-vectorization of eval_scalar, re-baseline | ✅ Done |
| 2 | Loop unrolling — eval_avx2_unroll2 + eval_avx2_unroll4, correctness tests | ✅ Done |
| 3 | Benchmark unrolled variants at all three sizes | ✅ Done |
| 4 | Software prefetching — eval_avx2_prefetch, correctness test + benchmark | ✅ Done |
| 5 | Popcount benchmark tiers — naive vs __builtin vs _mm_popcnt_u64 | ⬜ |

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

**Chunk 5 — Popcount tiers for blog narrative**

Benchmark three popcount implementations: naive bit-counting loop, `__builtin_popcountll`,
and `_mm_popcnt_u64`. Self-contained comparison for the developer blog post.

---

## Week 3 — Memory and parallelism

| Chunk | Deliverable | Status |
|-------|-------------|--------|
| TBD | Thread partitioning, NUMA-aware allocation, perf profiling on Akamai | ⬜ |

---

## Week 4 — Benchmarking and write-up

| Chunk | Deliverable | Status |
|-------|-------------|--------|
| TBD | CRoaring/DuckDB comparison, roofline model, blog post, CI matrix | ⬜ |
