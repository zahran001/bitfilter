# D4: Roofline Model — Analysis and Interpretation

## What a roofline chart is

A roofline model plots **achievable throughput** (Y-axis, GB/s) against
**operational intensity** (X-axis, ops/byte). It answers one question:
*is this workload bottlenecked by compute or by memory?*

The chart has two ceilings that form an inverted-L shape:
- **Horizontal ceiling:** memory bandwidth — the fastest the hardware can feed data
- **Diagonal ceiling:** compute throughput — the fastest the hardware can crunch ops

Every workload lands somewhere below both ceilings. If it's under the horizontal
line, it's **memory-bound** (the bus can't deliver data fast enough). If it's under
the diagonal, it's **compute-bound** (the CPU can't process fast enough).

For BitFilter, everything is deep in the memory-bound region (OI < 0.1 ops/byte).
The compute ceiling is irrelevant — we never come close to it.

---

## Why ops/byte, not FLOP/byte

Traditional roofline charts use FLOP/byte because most HPC workloads do
floating-point math. BitFilter does zero floating-point — it's all bitwise
AND/ANDN/OR and popcount. Using "FLOP" would be technically incorrect.

**Ops/byte** counts each bitwise operation per byte of data transferred. For
bitmap eval: one AND or ANDN per 8 bytes read ≈ **0.03 ops/byte**. For popcount:
one popcnt per 8 bytes ≈ **0.01 ops/byte**. Both values are extremely low,
which is the point — these kernels do almost no work per byte.

---

## Hardware ceilings (this machine)

| Parameter | Value | Source |
|-----------|-------|--------|
| Theoretical DRAM BW | 42.6 GB/s | DDR4-2660 × 2 channels × 8 bytes |
| Practical DRAM BW | 30–35 GB/s | DRAM refresh, page open/close, bus turnaround |
| Peak compute (bitwise) | ~288 Gops/s | 2 AVX2 ports × 32 bytes × ~4.5 GHz |

The practical ceiling is **not** a WSL2 artifact. It's a property of the DDR4
memory controller itself — bare metal Linux would hit the same wall. The gap
between 42.6 and 30–35 comes from DRAM refresh cycles, row buffer misses,
and the fact that 100% bus utilization is physically impossible.

---

## Measured data points

| Variant | Throughput (GB/s) | OI (ops/byte) | Bound |
|---------|------------------|---------------|-------|
| Eval scalar (auto-vec) | ~29 (actual DRAM) | ~0.03 | Memory |
| Eval AVX2+prefetch | ~29 (actual DRAM) | ~0.03 | Memory |
| Popcount ST | 17.1 | ~0.01 | Memory (not saturated) |
| Popcount MT (8T) | 31.3 | ~0.01 | Memory (saturated) |

---

## Key findings

### 1. Eval scalar and Eval AVX2+prefetch perform identically

Both sit at ~29 GB/s actual DRAM throughput. They are stacked on top of each
other on the chart. This is the single most important finding.

**Why:** The CPU is so fast at bitwise logic (even with scalar instructions) that
it finishes the math before RAM can deliver the next cache line. AVX2 processes
256 bits at a time instead of 64, but it doesn't matter — the bottleneck is the
memory bus, not the ALU. It's a rocket engine on a car stuck in traffic.

The compiler's auto-vectorization of the scalar path already generates SIMD
instructions (GCC `-O3 -march=native`). The "scalar" code isn't truly scalar at
the machine level — it's the compiler's own AVX2 codegen. Both paths issue
similar memory access patterns, so both hit the same wall.

**What this means:** For simple boolean bitmap queries, **scalar C++ compiled
with -O3 is already optimal.** The real optimization was identifying that you
can't go faster — the hardware is the limit.

### 2. Write-allocate makes the app-level numbers misleading

The code reports **23 GB/s** of application throughput (3 reads + 1 write =
4 × 62.5 MB / 10.7 ms). But the DRAM bus is actually moving **29 GB/s**.

**Why:** On x86, when you store to a cache line that isn't already in cache,
the CPU must first **read** that line from DRAM (a "Read For Ownership" / RFO),
then write to it, then eventually write it back. So every store to a cold line
costs one extra DRAM read that the application never asked for.

The actual DRAM traffic for eval:
- 3 bitmap reads: 3 × 62.5 MB = 187.5 MB
- 1 RFO (write-allocate): 1 × 62.5 MB = 62.5 MB  ← **hidden**
- 1 writeback: 1 × 62.5 MB = 62.5 MB
- **Total: 312.5 MB** (not 250 MB)
- 312.5 MB / 10.7 ms = **29.2 GB/s** actual bus traffic

This is 26% more than what the application "sees." And 29 GB/s is right at
the practical DRAM ceiling — which is why threading can't help. The bus is
already full.

### 3. Popcount tells the opposite story

Popcount is a **read-only** operation (scan a bitmap, count set bits, no output
bitmap). No writes means no write-allocate overhead — what the app reports is
what the bus actually moves.

**Popcount ST (17.1 GB/s):** A single thread running `_mm_popcnt_u64` in a loop
only achieves 40% of the theoretical peak. The core can't generate memory
requests fast enough to saturate the bus on its own. There is headroom.

**Popcount MT at 8 threads (31.3 GB/s):** Adding threads provides the
concurrency pressure to fill the memory controller's request queue. Eight cores
issuing parallel prefetch streams push the bus to 73% of theoretical — right
at the practical ceiling. This is real scaling (1.70x over ST).

**Why popcount scales but eval doesn't:**

| Property | Eval | Popcount |
|----------|------|----------|
| ST throughput | 29 GB/s (actual) | 17.1 GB/s |
| Headroom to ceiling | ~0% | ~60% |
| Write traffic | Yes (write-allocate) | None |
| Threading benefit | None | 1.70x at 8T |

Eval already saturates the bus at 1 thread because write-allocate inflates the
traffic by 26%. Popcount at 1 thread leaves room because reads are more
efficient (no bus turnaround between read and write phases, no RFO stalls).

### 4. Diminishing returns in popcount MT

| Threads | Throughput | Speedup |
|---------|-----------|---------|
| 1 | 17.1 GB/s | 1.00x |
| 2 | 29.1 GB/s | 1.68x |
| 4 | 30.5 GB/s | 1.64x |
| 8 | 31.3 GB/s | 1.70x |
| 12 | 30.8 GB/s | 1.58x |

Almost all the scaling comes from 1→2 threads. Going from 2T to 8T adds only
2 GB/s. At 12T, performance actually regresses due to thread spawn overhead
and memory controller contention. The bus saturates at 2–4 threads for
read-only traffic.

---

## How to read the chart

```
                    ┌──────────────────── Theoretical DRAM: 42.6 GB/s
                    │   ┌──────────────── Practical band: 30-35 GB/s
                    │   │
  Throughput (GB/s) │   │  ▲ Popcount MT (31.3) ← threading helped (read-only)
                    │   │  ◆ Eval AVX2 (29)     ← at the wall
                    │   │  ◆ Eval scalar (29)   ← also at the wall (same speed!)
                    │   │
                    │   │  ■ Popcount ST (17.1)  ← headroom exists
                    │
                    └────────────────────────────
                         0.01    0.03          Operational Intensity (ops/byte)
                                               (all points are far left = memory-bound)
```

Everything clusters in the bottom-left corner. This is a deeply memory-bound
workload. The compute diagonal is irrelevant — the CPU could do 10x more math
per byte and still be waiting for DRAM.

---

## Reproducing the chart

```bash
# Requires: python3, matplotlib (sudo apt install python3-matplotlib)
cd /home/zahran1/projects/bitfilter
python3 scripts/roofline.py
# Output: site/img/roofline.svg and site/img/roofline.png
```

The script (`scripts/roofline.py`) hardcodes the measured values from Week 3
verification runs. If numbers change, update the `variants` list and hardware
constants at the top of the script.

---

## Portfolio angle

1. **Identifying the bottleneck.** The workload is memory-bound. No amount of
   SIMD cleverness or threading will push eval past ~29 GB/s on this hardware.

2. **Write-allocate awareness.** Most developers report application-level GB/s
   and never realize their bus is carrying 26% more traffic. Understanding the
   hidden RFO traffic explains why eval can't scale with threads.

3. **Knowing when to stop optimizing.** Scalar C++ at -O3 already reaches the
   memory wall for eval. The "optimization" is proving that no further
   optimization is possible — and explaining why.

4. **Knowing when threading helps.** Popcount scales because a single thread
   can't saturate read-only bandwidth. Eval doesn't scale because write-allocate
   pushes a single thread to the ceiling. Same hardware, opposite conclusions —
   the difference is the memory access pattern.

5. **Mechanical sympathy.** The entire analysis flows from understanding cache
   lines, write-allocate, DRAM bandwidth limits, and memory controller queuing.
   This is the CPU architecture intuition that matters for systems programming.
