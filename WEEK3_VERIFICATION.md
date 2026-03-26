# Week 3 — Verification Log

Systematic verification of MT benchmark results before trusting them.

---

## V1: 1T MT vs standalone single-threaded gap

**Question:** The 1T MT wrapper reports 13.3 ms, but standalone `eval_avx2_prefetch`
ran at 11.3 ms in the same session. The MT fallback path calls `eval_avx2_prefetch`
directly — why the 18% gap?

**Hypotheses:**
- (a) Benchmark variance between runs (thermal, background load)
- (b) Google Benchmark CPU time vs wall time confusion
- (c) Different iteration counts causing different warm-up behavior

**Method:** Run both benchmarks back-to-back in a single invocation, compare wall time.

**Result:** TBD

---

## V2: Wall time vs CPU time — which metric are we reporting?

**Question:** Google Benchmark defaults to CPU time. For MT workloads, CPU time sums
across threads, which inflates the number. We need wall time for latency claims and
CPU time for efficiency claims. Are we consistent?

**Method:** Re-run MT benchmark with `--benchmark_counters_tabular=true` and inspect
both Time and CPU columns. Confirm `SetBytesProcessed` divides by wall time (it does
by default in Google Benchmark — verify).

**Result:** TBD

---

## V3: Are threads actually running in parallel?

**Question:** We assume over-partitioned dispatch works, but haven't proven threads
claim chunks concurrently. A pathological case: all chunks claimed by one thread
before others start.

**Method:** Add a temporary diagnostic build that counts chunks per thread. Run at
500M / 12 threads, print distribution. Remove diagnostic after verification.

**Result:** TBD

---

## V4: Throughput calculation — write-allocate traffic

**Question:** `SetBytesProcessed` counts `n_words * 4 * 8` (3 reads + 1 write).
On x86, stores to cold cache lines trigger write-allocate: the CPU reads the line
from DRAM before writing. So actual DRAM traffic is closer to 5 × bitmap_size,
not 4×. Are our GB/s numbers overstated?

**Method:** Reason about write-allocate impact. If the result buffer is cold (likely
at 500M), actual DRAM traffic = 3 reads + 1 read-for-ownership + 1 write = 5×.
Recalculate effective bandwidth. Compare with popcount (read-only, no write-allocate).

**Result:** TBD

---

## V5: Warm vs cold — iteration count check

**Question:** Week 2 showed cold-CPU artifacts (7.5 GB/s was bogus). Are Week 3
benchmarks running enough iterations to be past warm-up?

**Method:** Check iteration counts from benchmark output. Google Benchmark auto-scales
iterations. If count > 10, warm-up is likely sufficient. Also check if first-iteration
effects are visible.

**Result:** TBD

---

## V6: Apples-to-apples baseline

**Question:** The scaling ratios (1.22x at 2T, etc.) are computed against the MT
wrapper's own 1T number, not the standalone single-threaded benchmark. Is this fair?

**Method:** Run all variants in a single benchmark invocation:
- `EvalFixture/Avx2Prefetch/500000000` (standalone ST)
- `MtEvalFixture/Avx2PrefetchMT/500000000/{1,2,4,8,12}` (MT wrapper)

Compute scaling ratios against the standalone ST number.

**Result:** TBD
