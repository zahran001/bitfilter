# BitFilter — Verification Guide

## The anomaly

Week 1 scalar showed **7.5 GB/s** at 500M users.
Week 2 de-vectorized scalar showed **16.2 GB/s** at 500M users.

Two competing explanations:

1. **Thermal/cold artifact** — the W1 run was noisy
2. **Auto-vectorization genuinely hurt** — GCC's vectorized loop had more instruction overhead per word at DRAM scale

This guide resolves which explanation is correct.

---

## Prerequisites

```bash
cd /home/zahran1/projects/bitfilter
```

Confirm tests pass before any experiments:

```bash
cmake --build build -j$(nproc) && ./build/test_correctness
```

---

## Step 1: Inspect the assembly (definitive, zero-cost)

**Priority: Do first.** This costs nothing and is definitive. If `ymm` registers
appear when the attribute is removed, auto-vectorization is confirmed as real.

First, check the **current** (protected) assembly by compiling directly to
assembly text — this is more reliable than `objdump` (which can miss inlined
symbols or require C++ name mangling):

```bash
g++ -O3 -march=native -mavx2 -std=c++20 \
    -Iinclude -S src/query_eval_scalar.cpp -o /tmp/scalar_protected.s

grep -E "ymm|vpand|vmovdqa" /tmp/scalar_protected.s
```

You should see **no output** — no `ymm` registers, no vector instructions.
The `no-tree-vectorize` attribute is keeping GCC scalar.

Now temporarily remove the protection and check again:

```bash
cp src/query_eval_scalar.cpp src/query_eval_scalar.cpp.bak

sed -i 's/^__attribute__((optimize("no-tree-vectorize")))/\/\/ __attribute__((optimize("no-tree-vectorize")))/' \
    src/query_eval_scalar.cpp

g++ -O3 -march=native -mavx2 -std=c++20 \
    -Iinclude -S src/query_eval_scalar.cpp -o /tmp/scalar_autovec.s

grep -E "ymm|vpand|vmovdqa" /tmp/scalar_autovec.s
```

**What to look for:**

| Assembly output | Meaning |
|-----------------|---------|
| Lines with `ymm`, `vpand`, `vpandn`, `vmovdqa` | GCC auto-vectorized to AVX2 — auto-vectorization is real |
| No output from grep | GCC did NOT auto-vectorize (unlikely at `-O3 -march=native`) |

You can also diff the two for a full picture:

```bash
diff /tmp/scalar_protected.s /tmp/scalar_autovec.s
```

If you see `ymm` instructions, auto-vectorization is confirmed. Proceed to Step 2
to measure how much it actually costs.

---

## Step 2: Benchmark auto-vectorized scalar (measure the impact)

**Priority: Do second.** The source from Step 1 still has auto-vectorization enabled.
Rebuild the full binary and benchmark:

```bash
cmake --build build -j$(nproc)
./build/bench --benchmark_filter=Scalar/500000000 --benchmark_repetitions=5
```

**Interpretation:**

| Result | Meaning |
|--------|---------|
| ~16 GB/s | W1's 7.5 GB/s was a cold-CPU artifact. Auto-vectorization doesn't hurt at DRAM scale. |
| ~11–14 GB/s | Mixed: some vectorization overhead + some measurement noise. Run Step 3 to separate the two. |
| ~7–9 GB/s | The compiler's vectorized loop genuinely underperforms at DRAM scale due to instruction overhead. |

---

## Step 3: Rule out thermal throttling (only if Step 2 gives surprising results)

**Priority: Do only if Step 2 shows ~7–9 GB/s.** If Step 2 already shows ~16 GB/s,
skip to Step 4.

### 3a. Monitor CPU frequency

Open a **second terminal**:

```bash
watch -n 0.5 "grep 'cpu MHz' /proc/cpuinfo | head -4"
```

In the **first terminal**, re-run the benchmark and watch for frequency ramp-up:

```bash
./build/bench --benchmark_filter=Scalar/500000000
```

| Observation | Meaning |
|-------------|---------|
| Starts ~800 MHz, climbs to ~3500+ MHz | Thermal ramp-up — first iterations are artificially slow |
| Stable ~3500+ MHz throughout | CPU was already warm, throttling is not the issue |

### 3b. Force warmup with back-to-back runs

```bash
# Run 1: warms the CPU (throwaway)
./build/bench --benchmark_filter=Scalar/500000000 --benchmark_min_time=2

# Run 2: real measurement
./build/bench --benchmark_filter=Scalar/500000000 --benchmark_repetitions=5 \
    --benchmark_report_aggregates_only=true
```

If Run 2 shows significantly higher GB/s than the first attempt in Step 2,
cold-start was the culprit.

---

## Step 4: Restore the codebase (always do this)

```bash
cp src/query_eval_scalar.cpp.bak src/query_eval_scalar.cpp
cmake --build build -j$(nproc)
rm src/query_eval_scalar.cpp.bak

# Confirm correctness
./build/test_correctness
```

Never leave the codebase in a modified state.

---

## Decision matrix

| Step 1 result | Step 2 result | Step 3 result | Conclusion |
|---------------|---------------|---------------|------------|
| `ymm` present | ~16 GB/s | (skipped) | Auto-vectorization is real but doesn't hurt. W1 was a cold/throttled run. |
| `ymm` present | ~7–9 GB/s | Warmup fixes it | Auto-vectorization is real and W1 was cold/throttled. |
| `ymm` present | ~7–9 GB/s | Warmup doesn't fix | Auto-vectorization genuinely hurts at DRAM scale — more instructions per word. |
| No `ymm` | (any) | (any) | GCC wasn't auto-vectorizing. The 7.5 GB/s was purely a measurement artifact. |

---

## Results (2026-03-22)

**Step 1:** `ymm` registers confirmed present when `no-tree-vectorize` is removed.
GCC emits `vpxor`, `vpand`, `vmovdqu` with `%ymm` registers — full AVX2 auto-vectorization.
With the attribute in place, grep returns nothing — pure scalar.

**Step 2:** Auto-vectorized scalar at 500M users = **18.0 GB/s median** (13.0 ms CPU time,
5 reps, 6% CV). This matches the de-vectorized scalar baseline (~16–17 GB/s).

**Step 3:** Skipped — Step 2 already showed ~18 GB/s.

**Step 4:** Codebase restored, `no-tree-vectorize` attribute back in place, all 16 tests passing.

### Conclusion

**Row 1 of the decision matrix.** W1's 7.5 GB/s was a cold-CPU / thermal-throttle artifact.
Auto-vectorization is real (GCC does emit AVX2) but irrelevant at DRAM scale — the CPU is
memory-bound either way, so scalar vs SIMD instruction mix doesn't change throughput.
The `no-tree-vectorize` attribute remains correct for its original purpose: ensuring the
scalar path uses genuinely different instructions than the AVX2 path, so memcmp correctness
tests can catch operand-order bugs.
