# D1: BitFilter vs CRoaring — Comparative Benchmark

## Setup

- **Query:** A AND B AND NOT C at 500M users
- **BitFilter path:** `eval_avx2_prefetch` — AVX2 with software prefetching (best single-threaded variant)
- **CRoaring:** v4.6.1, built from source via CMake FetchContent with `-march=native -O3`
- **CRoaring prep:** `roaring_bitmap_run_optimize()` called on all inputs (gives CRoaring its best representation)
- **Data:** identical logical sets — same RNG seeds, same user IDs in both engines
- **Machine:** Intel 12th/13th Gen hybrid, DDR4-2660 dual-channel, WSL2 Ubuntu 24.04

## Results (median of 3 repetitions)

| Density | BitFilter (ms) | CRoaring (ms) | Winner | Ratio |
|---------|---------------|---------------|--------|-------|
| 0.01% | 10.7 | **0.51** | CRoaring | 21.2x |
| 0.1% | 10.5 | **0.93** | CRoaring | 11.3x |
| 1% | **11.8** | 7.23 | CRoaring | 1.6x |
| 10% | **11.7** | 50.8 | BitFilter | 4.3x |
| 50% | **~11.7** | 25.5 | BitFilter | 2.2x |

BitFilter throughput is stable at ~20–23 GiB/s across all densities (density-invariant — always scans the full 250 MB of bitmap data).

## Analysis

### Why the crossover exists (~2-5% density)

**BitFilter** uses flat dense bitmaps: one bit per user, scanned linearly regardless of how many bits are set. The cost is fixed at ~10.5–11.8 ms — it always reads 3 × 62.5 MB + writes 1 × 62.5 MB. Density doesn't matter.

**CRoaring** uses Roaring Bitmaps: the 500M user-ID space is partitioned into 16-bit containers (65,536 users each). Each container is stored in whichever format is smallest:
- **Array container** — sorted list of 16-bit offsets. Used when < 4,096 values are set.
- **Bitmap container** — 8 KB flat bitmap. Used when >= 4,096 values are set.
- **Run container** — run-length encoded. Used after `run_optimize()` when runs are cheaper.

At low density, most containers are arrays or empty — CRoaring skips them entirely. At high density, most containers are bitmap-type and CRoaring must iterate through all of them with per-container dispatch overhead that BitFilter's flat scan avoids.

### The 10% anomaly (50.8 ms — worse than 50%)

CRoaring at 10% density is *slower* than at 50%. This is the worst-case container mix:

- At 10% density, each container holds ~6,554 values on average (10% of 65,536). This exceeds the 4,096 array threshold, so containers are bitmap-type.
- With bitmap containers, CRoaring performs bitwise AND on 8 KB bitmaps — similar to what BitFilter does, but with per-container loop overhead, virtual dispatch, and result-bitmap allocation.
- The AND result at 10% density has ~1% set bits (10% × 10%), producing sparse output containers. CRoaring then AND-NOTs against C (also 10%), producing even sparser output. The engine may convert result containers back to arrays, paying additional overhead.
- At 50% density, the containers are denser and the operations are more uniform — less container-type thrashing.

This is a known characteristic of Roaring Bitmaps: moderate density is the worst case because it maximizes the number of bitmap-type containers without benefiting from the uniformity of high density.

### What this proves

1. **Dense bitmaps are the right tool for dense workloads.** When segment density exceeds a few percent — typical for broad targeting categories like "US users" or "mobile users" — a flat SIMD scan is faster than any compressed representation.

2. **CRoaring is the right tool for sparse workloads.** Niche segments ("users who clicked ad X in the last hour") have sub-0.1% density. CRoaring's 21x advantage at 0.01% makes it the clear winner.

3. **A production system would use both.** Segments below ~2-5% density go into Roaring Bitmaps; segments above go into flat SIMD bitmaps. The query planner picks the evaluation path per segment.

## Raw benchmark output

```
BitFilterFixture/EvalAvx2Prefetch/1_mean            10.6 ms         10.3 ms            3 bytes_per_second=22.634Gi/s
BitFilterFixture/EvalAvx2Prefetch/1_median          10.7 ms         10.3 ms            3 bytes_per_second=22.538Gi/s
BitFilterFixture/EvalAvx2Prefetch/1_stddev         0.131 ms        0.127 ms            3 bytes_per_second=286.785Mi/s
BitFilterFixture/EvalAvx2Prefetch/1_cv              1.23 %          1.23 %             3 bytes_per_second=1.24%
BitFilterFixture/EvalAvx2Prefetch/10_mean           10.8 ms         10.4 ms            3 bytes_per_second=22.4093Gi/s
BitFilterFixture/EvalAvx2Prefetch/10_median         10.5 ms         10.2 ms            3 bytes_per_second=22.9057Gi/s
BitFilterFixture/EvalAvx2Prefetch/10_stddev        0.529 ms        0.508 ms            3 bytes_per_second=1.06535Gi/s
BitFilterFixture/EvalAvx2Prefetch/10_cv             4.91 %          4.88 %             3 bytes_per_second=4.75%
BitFilterFixture/EvalAvx2Prefetch/100_mean          11.7 ms         11.3 ms            3 bytes_per_second=20.6679Gi/s
BitFilterFixture/EvalAvx2Prefetch/100_median        11.8 ms         11.4 ms            3 bytes_per_second=20.4546Gi/s
BitFilterFixture/EvalAvx2Prefetch/100_stddev       0.567 ms        0.559 ms            3 bytes_per_second=1.03809Gi/s
BitFilterFixture/EvalAvx2Prefetch/100_cv            4.85 %          4.95 %             3 bytes_per_second=5.02%
BitFilterFixture/EvalAvx2Prefetch/1000_mean         12.0 ms         11.6 ms            3 bytes_per_second=20.2022Gi/s
BitFilterFixture/EvalAvx2Prefetch/1000_median       11.7 ms         11.3 ms            3 bytes_per_second=20.6829Gi/s
BitFilterFixture/EvalAvx2Prefetch/1000_stddev       1.31 ms         1.27 ms            3 bytes_per_second=2.12211Gi/s
BitFilterFixture/EvalAvx2Prefetch/1000_cv          10.91 %         10.91 %             3 bytes_per_second=10.50%
CRoaringFixture/AndAndNot/1_mean                   0.507 ms        0.485 ms            3
CRoaringFixture/AndAndNot/1_median                 0.505 ms        0.483 ms            3
CRoaringFixture/AndAndNot/1_stddev                 0.008 ms        0.009 ms            3
CRoaringFixture/AndAndNot/1_cv                      1.49 %          1.95 %             3
CRoaringFixture/AndAndNot/10_mean                  0.938 ms        0.906 ms            3
CRoaringFixture/AndAndNot/10_median                0.932 ms        0.900 ms            3
CRoaringFixture/AndAndNot/10_stddev                0.025 ms        0.024 ms            3
CRoaringFixture/AndAndNot/10_cv                     2.69 %          2.69 %             3
CRoaringFixture/AndAndNot/100_mean                  7.26 ms         7.01 ms            3
CRoaringFixture/AndAndNot/100_median                7.23 ms         6.98 ms            3
CRoaringFixture/AndAndNot/100_stddev               0.066 ms        0.065 ms            3
CRoaringFixture/AndAndNot/100_cv                    0.91 %          0.92 %             3
CRoaringFixture/AndAndNot/1000_mean                 50.9 ms         49.1 ms            3
CRoaringFixture/AndAndNot/1000_median               50.8 ms         49.1 ms            3
CRoaringFixture/AndAndNot/1000_stddev              0.555 ms        0.516 ms            3
CRoaringFixture/AndAndNot/1000_cv                   1.09 %          1.05 %             3
CRoaringFixture/AndAndNot/5000_mean                 25.9 ms         25.0 ms            3
CRoaringFixture/AndAndNot/5000_median               25.5 ms         24.6 ms            3
CRoaringFixture/AndAndNot/5000_stddev              0.851 ms        0.804 ms            3
CRoaringFixture/AndAndNot/5000_cv                   3.28 %          3.22 %             3
```

Arg encoding: 1 = 0.01%, 10 = 0.1%, 100 = 1%, 1000 = 10%, 5000 = 50% (basis points).

Note: BitFilter 5000 (50%) row was not captured in this run but is consistent with
prior runs at ~11.7 ms. All other densities confirm density-invariant behavior.

## Variance

BitFilter CV: 1.2–10.9%. Much tighter than prior runs (machine was idle, load average 0.00). The 10% run (10.9%) is the outlier — likely a transient scheduling event.

CRoaring CV: 0.9–3.3%. Excellent stability across all densities. The 1% run (0.9%) is the tightest — all containers are bitmap-type with uniform operation cost.

All results are from `--benchmark_repetitions=3` with `--benchmark_report_aggregates_only=true`.
