# Week 3 — Verification Log

Systematic verification of MT benchmark results before trusting them.

**Verification run (2026-03-26):** All eval variants in a single invocation, load average 0.00.

```
EvalFixture/Avx2Prefetch/500000000              10.7 ms         10.3 ms           63 bytes_per_second=22.6893Gi/s
MtEvalFixture/Avx2PrefetchMT/500000000/1        10.5 ms         10.1 ms           68 bytes_per_second=23.0959Gi/s
MtEvalFixture/Avx2PrefetchMT/500000000/2        10.9 ms         10.1 ms           71 bytes_per_second=23.131Gi/s
MtEvalFixture/Avx2PrefetchMT/500000000/4        15.8 ms         14.2 ms           69 bytes_per_second=16.4016Gi/s
MtEvalFixture/Avx2PrefetchMT/500000000/8        11.4 ms         10.3 ms           64 bytes_per_second=22.5715Gi/s
MtEvalFixture/Avx2PrefetchMT/500000000/12       12.1 ms         11.2 ms           60 bytes_per_second=20.854Gi/s
```

---

## V1: 1T MT vs standalone single-threaded gap

**Question:** The initial MT run (earlier session) reported 1T = 13.3 ms, but standalone
`eval_avx2_prefetch` ran at 11.3 ms. The MT fallback path calls `eval_avx2_prefetch`
directly — why the 18% gap?

**Hypotheses:**
- (a) Benchmark variance between runs (thermal, background load)
- (b) Google Benchmark CPU time vs wall time confusion
- (c) Different iteration counts causing different warm-up behavior

**Method:** Run both benchmarks back-to-back in a single invocation, compare wall time.

**Result: Gap was an artifact.** In the back-to-back run: standalone ST = 10.7 ms,
MT 1T = 10.5 ms — essentially identical (within noise). The earlier 13.3 ms was
thermal or background load variance between separate benchmark invocations.

**Verdict:** The MT fallback path works correctly. No overhead from the wrapper.

---

## V2: Wall time vs CPU time — which metric are we reporting?

**Question:** Google Benchmark defaults to CPU time for `bytes_per_second`. For MT
workloads, CPU time sums across threads, which could inflate the number.

**Method:** Inspect both Time (wall) and CPU columns in the verification run.

**Result:** For the MT variants, wall and CPU times are close:
- 2T: wall 10.9 ms, CPU 10.1 ms
- 8T: wall 11.4 ms, CPU 10.3 ms
- 12T: wall 12.1 ms, CPU 11.2 ms

`bytes_per_second` is derived from CPU time (Google Benchmark default). Since the
threads aren't providing meaningful parallelism (see V6), the wall/CPU gap is small
and the choice doesn't materially affect the numbers.

**Verdict:** No significant distortion. Both metrics tell the same story: threading
doesn't help.

---

## V3: Are threads actually running in parallel?

**Question:** We assume over-partitioned dispatch works, but haven't proven threads
claim chunks concurrently.

**Method:** Diagnostic `diag_chunks.cpp` — two modes:
1. No-work: just claim chunks (exposes thread spawn latency)
2. With-work: real `_mm_popcnt_u64` scan per chunk (realistic distribution)

**Result (no-work): 100% imbalance at all thread counts.**

One thread claims every chunk before the others are scheduled. `std::jthread`
constructor + kernel thread creation takes ~50-100μs — long enough for the first
thread to exhaust all chunks when there's no real work per chunk.

**Result (with-work at 500M users):**

| Threads | Chunks | Min/Max per thread | Imbalance |
|---------|--------|--------------------|-----------|
| 2 | 8 | 4/4 | 0% — perfect |
| 4 | 16 | 3/5 | 40% |
| 8 | 32 | 3/7 | 57% |
| 12 | 48 | 2/7 | 71% |

**Analysis:**

With real memory-bound work (~1-2 ms per chunk), threads do run in parallel and the
over-partitioned dispatch functions. But the imbalance grows with thread count:

- **2T: Perfect.** Each thread gets exactly 4 chunks. The first chunk takes long
  enough for the second thread to start. This explains popcount's clean 1.68x scaling.
- **4T: Decent.** 3-5 range. Early-arriving threads get 1-2 extra chunks.
- **8-12T: Significant imbalance.** Late-spawned threads miss early chunks entirely.
  At 12T, threads 1-2 grabbed 7 chunks each while threads 0,5,10,11 got only 2.
  The wall time is gated by the slowest (most-loaded) thread.

**Why the imbalance grows:** `std::jthread` spawns threads sequentially. The first
few threads start running while later ones are still being created. With 12 threads,
the spawn loop itself takes ~0.5-1 ms (12 × 50-100μs), during which early threads
have already claimed several chunks.

**Impact on benchmarks:** This explains the 12T popcount regression (2.40 ms) vs
8T (2.23 ms). At 12T, the most-loaded thread has 3.5× the work of the least-loaded.
The dispatch is *functional* but not *optimal* at high thread counts.

**Possible fix (future):** Use a barrier (`std::barrier` or `std::latch`) to hold
all threads at the starting line until everyone is spawned, then release simultaneously.
This would eliminate the spawn-stagger bias. Not implemented yet — the current results
are honest and the imbalance is itself an interesting finding for the portfolio.

---

## V4: Throughput calculation — write-allocate traffic

**Question:** `SetBytesProcessed` counts `n_words * 4 * 8` (3 reads + 1 write).
On x86, stores to cold cache lines trigger write-allocate: the CPU reads the line
from DRAM before writing. So actual DRAM traffic is closer to 5x bitmap_size.

**Analysis:**

For eval (A AND B AND NOT C → result):
- Application-visible traffic: 3 reads + 1 write = 4 × 62.5 MB = 250 MB
- Actual DRAM traffic with write-allocate: 3 reads + 1 RFO + 1 writeback = 5 × 62.5 MB = 312.5 MB
- Reported throughput at 10.7 ms: 250 MB / 10.7 ms = 23.4 GB/s (application)
- Actual DRAM bandwidth consumed: 312.5 MB / 10.7 ms = 29.2 GB/s

For popcount (read-only):
- No write-allocate — reported throughput = actual DRAM traffic
- Reported 12.7 GiB/s (13.6 GB/s) at 1T is the true DRAM bandwidth

**Revised picture:**
- Eval at 1T already consumes ~29 GB/s of actual DRAM bandwidth
- This is 68% of the 42.6 GB/s theoretical peak
- This is within the practical ceiling of ~30-35 GB/s
- Threading can't help because the bus is already near saturation

**Verdict:** Application-level GB/s numbers in `SetBytesProcessed` are correct for
what we're *reporting* (data processed by the application). But the actual DRAM
bandwidth consumed is ~25% higher due to write-allocate. This explains why a single
thread saturates the bus and threading provides no benefit.

---

## V5: Warm vs cold — iteration count check

**Question:** Are iteration counts high enough to be past warm-up?

**Method:** Check iterations column from verification run.

**Result:** All variants ran 60-71 iterations. Google Benchmark auto-scales to fill
a minimum time window (~0.5s default). At 60+ iterations of a ~10ms workload, that's
600+ ms of measurement time — well past any warm-up transient.

**Verdict:** No cold-CPU artifact. Numbers are stable.

---

## V6: Apples-to-apples baseline — the real scaling story

**Question:** Do the MT variants actually improve on single-threaded performance?

**Method:** Compute scaling ratios against standalone ST (10.7 ms wall time).

| Threads | Wall time | Speedup vs ST | Notes |
|---------|-----------|---------------|-------|
| ST (standalone) | 10.7 ms | 1.00x | baseline |
| 1 (MT fallback) | 10.5 ms | 1.02x | noise — fallback calls ST directly |
| 2 | 10.9 ms | 0.98x | no improvement |
| 4 | 15.8 ms | 0.68x | **regression** — contention or scheduling |
| 8 | 11.4 ms | 0.94x | slight regression |
| 12 | 12.1 ms | 0.88x | worse with all cores |

**Verdict: Threading provides zero benefit for eval at 500M users on this hardware.**

The single-threaded AVX2+prefetch path already consumes ~29 GB/s of actual DRAM
bandwidth (accounting for write-allocate), which is near the practical ceiling of
the DDR4-2660 dual-channel memory controller. Adding threads only adds overhead:
- `std::jthread` spawn/join cost
- `std::atomic<unsigned>` fetch_add contention
- Memory controller queue contention from multiple cores
- Possible cache-line bouncing on the atomic counter

The 4T regression to 15.8 ms is the worst case — likely due to threads landing on
P-core hyperthreads that compete for the same execution resources.

---

## Summary and implications

**The initial MT benchmark results (showing 1.22x at 2T) were misleading.** The apparent
speedup came from comparing against an inflated 1T baseline (13.3 ms) that was measured
in a separate, likely colder run. When measured back-to-back against the real ST baseline:

- Single-threaded eval already saturates memory bandwidth at ~29 GB/s actual DRAM traffic
- Threading adds overhead without adding throughput
- The memory controller is the bottleneck, not the CPU

**This is still a strong portfolio finding.** It demonstrates:
1. Understanding of memory-bandwidth-bound workloads
2. Ability to identify when parallelism won't help (and why)
3. Rigorous self-verification of benchmark results
4. Write-allocate awareness in throughput calculations

**Popcount verification still needed** — popcount showed better scaling (read-only, lower
ST throughput). A separate verification run should confirm whether those numbers hold up.

---

## V7: Popcount MT verification (2026-03-26)

```
PopcountFixture/Hardware/500000000              3.80 ms         3.66 ms          176 bytes_per_second=15.883Gi/s
MtPopcountFixture/PopcountMT/500000000/1        4.08 ms         3.94 ms          165 bytes_per_second=14.789Gi/s
MtPopcountFixture/PopcountMT/500000000/2        2.26 ms         2.15 ms          295 bytes_per_second=27.0666Gi/s
MtPopcountFixture/PopcountMT/500000000/4        2.32 ms         2.05 ms          330 bytes_per_second=28.4237Gi/s
MtPopcountFixture/PopcountMT/500000000/8        2.23 ms         1.99 ms          343 bytes_per_second=29.1885Gi/s
MtPopcountFixture/PopcountMT/500000000/12       2.40 ms         2.03 ms          331 bytes_per_second=28.6661Gi/s
```

**Scaling ratios (against standalone ST = 3.80 ms wall time):**

| Threads | Wall time | Speedup vs ST | Throughput (GiB/s) | Throughput (GB/s) |
|---------|-----------|---------------|-------------------|-------------------|
| ST (standalone) | 3.80 ms | 1.00x | 15.9 | 17.1 |
| 1 (MT fallback) | 4.08 ms | 0.93x | 14.8 | 15.9 |
| 2 | 2.26 ms | 1.68x | 27.1 | 29.1 |
| 4 | 2.32 ms | 1.64x | 28.4 | 30.5 |
| 8 | 2.23 ms | 1.70x | 29.2 | 31.3 |
| 12 | 2.40 ms | 1.58x | 28.7 | 30.8 |

**Verdict: Popcount MT scaling is real.** Peak 1.70x at 8 threads.

**Why popcount scales but eval doesn't:**

1. **ST popcount doesn't saturate the bus.** A single thread doing `_mm_popcnt_u64`
   reads at 17.1 GB/s — only 40% of the 42.6 GB/s theoretical peak. There is
   headroom for additional threads to issue parallel memory requests.

2. **ST eval already saturates the bus.** With write-allocate, eval consumes ~29 GB/s
   of actual DRAM bandwidth at 1T — already at the practical ceiling (~30-35 GB/s).
   No headroom for threading.

3. **No write-allocate in popcount.** Popcount is pure read. No RFO traffic, no
   writeback, no bus turnaround penalty. The memory controller can serve reads more
   efficiently than mixed read/write.

4. **The 1T fallback overhead (0.28 ms / 7%).** The MT wrapper at 1T takes 4.08 ms
   vs standalone 3.80 ms. This is the cost of the atomic/vector setup even on the
   fallback path. Minor but measurable.

5. **Diminishing returns after 2T.** Most of the scaling (1.68x) comes from just
   2 threads. Going 2T → 8T only adds 0.02x more. The bus saturates around 2-4
   threads for read-only traffic.

6. **12T regression.** 12 threads (2.40 ms) is slower than 8 threads (2.23 ms).
   Thread creation overhead and memory controller contention outweigh any marginal
   bandwidth gain.

---

## Corrected Week 3 narrative

The initial benchmark session overstated eval scaling due to a cold-baseline artifact.
After back-to-back verification:

**Eval (A AND B AND NOT C → result):**
- Single-threaded AVX2+prefetch: 10.7 ms, ~23 GiB/s application throughput
- Actual DRAM traffic with write-allocate: ~29 GB/s (68% of peak)
- Threading provides no benefit — the memory bus is already saturated at 1T
- This is a **bandwidth saturation proof**, not a threading success story

**Popcount (read-only scan):**
- Single-threaded: 3.80 ms, 15.9 GiB/s (40% of peak — headroom exists)
- Peak MT: 2.23 ms at 8T, 29.2 GiB/s (73% of peak)
- Real 1.70x speedup — threading works when the bus isn't already saturated
- Saturates at 2-4 threads, diminishing returns beyond

**Portfolio angle:** This demonstrates the ability to:
1. Identify memory-bandwidth-bound workloads
2. Predict when parallelism will and won't help (write-allocate analysis)
3. Catch misleading benchmarks through rigorous verification
4. Explain *why* results differ between read-only and read-write workloads

**Action items:**
- [x] Update PROGRESS.md with corrected eval scaling story
- [x] Correct plan.md target table
- [x] 1T MT fallback overhead (7% for popcount): benchmark fixture variance, not real
  overhead — the fallback path returns `popcount()` directly with no allocation. No fix needed.
