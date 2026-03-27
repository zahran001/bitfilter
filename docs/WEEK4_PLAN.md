# Week 4 — Benchmarking, CI, and Write-up

## Goals

1. Benchmark BitFilter against CRoaring, DuckDB, and SQLite
2. Build a roofline model from existing data
3. Set up GitHub Actions CI (x86 AVX2 + ARM SVE)
4. Publish a GitHub Pages portfolio site (replaces a traditional blog post)

---

## Deliverables

### D1: CRoaring comparative benchmark

**Why:** CRoaring (Roaring Bitmaps) is the industry standard for compressed bitmap
operations. BitFilter uses dense bitmaps — CRoaring uses hybrid compressed bitmaps.
The comparison shows where each approach wins.

**Expected result:** BitFilter dominates at high segment density (>1%) where bitmaps
are nearly full and compression adds overhead. CRoaring wins at very low density
(<0.1%) where most containers compress to run-length or array form.

**Steps:**
- [x] Add CRoaring v4.6.1 via CMake FetchContent (source build with `-march=native`)
- [x] Write `bench/bench_croaring.cpp`: same A AND B AND NOT C query at 500M users
- [x] Run at multiple densities: 0.01%, 0.1%, 1%, 10%, 50%
- [x] Record latency and throughput for both engines at each density → see `docs/WEEK4_CROARING.md`
- [x] Add CRoaring benchmark target to CMakeLists.txt

### D2: DuckDB comparative benchmark

**Why:** DuckDB is a fast columnar SQL engine. Running the equivalent boolean query
shows the overhead of a general-purpose query engine vs a purpose-built bitmap engine.

**Steps:**
- [x] ~~Download DuckDB C++ amalgamation~~ Use prebuilt `libduckdb.so` v1.5.1 (amalgamation OOMs at <12 GB RAM)
- [x] Write `bench/bench_duckdb.cpp`: create in-memory table with boolean columns
      a, b, c for 500M rows, run `SELECT COUNT(*) WHERE a AND b AND NOT c`
- [x] Record query time at 10% and 50% density → see `docs/WEEK4_DUCKDB.md`
- [x] Add DuckDB benchmark target to CMakeLists.txt (prebuilt .so — no long compile)
- [x] Note: DuckDB uses its own SIMD internally — the point is abstraction overhead

### D3: SQLite comparative benchmark

**Why:** SQLite is a row-store. This is the "why bitmaps matter" baseline — shows
the cost of row-by-row evaluation at scale.

**Steps:**
- [ ] Write `bench/bench_sqlite.cpp` using SQLite C API: create in-memory table,
      populate 50M rows, run the equivalent boolean query
- [ ] Record query time, extrapolate to 500M (linear scaling for full-table scan)
- [ ] Site hook: "SQLite took X min for 50M; BitFilter took 11 ms for 500M"

### D4: Roofline model

**Why:** A roofline plot is the standard way to show whether a workload is
compute-bound or memory-bound, and how close it is to the hardware ceiling.

**Data already collected (Week 3):**

| Variant | Throughput (GB/s) | Operational intensity | Bound |
|---------|------------------|-----------------------|-------|
| Eval scalar (de-vec) | 16.7 | ~0.03 ops/byte | Memory |
| Eval AVX2+prefetch | ~29 (actual DRAM) | ~0.03 ops/byte | Memory |
| Popcount ST | 17.1 | ~0.01 ops/byte | Memory |
| Popcount MT (8T) | 31.3 | ~0.01 ops/byte | Memory |

Hardware ceilings:
- Theoretical DRAM BW: 42.6 GB/s (DDR4-2660 dual-channel)
- Practical DRAM BW: ~30-35 GB/s
- Compute peak: far above anything this workload needs

**Steps:**
- [ ] Write a Python script (`scripts/roofline.py`) to generate the roofline chart
- [ ] Plot all variants as points on the chart
- [ ] Export as SVG/PNG for the GitHub Pages site
- [ ] Include write-allocate annotation (why eval's actual BW is higher than reported)

### D5: GitHub Actions CI

**Why:** Proves the code builds and passes tests on both target architectures.

**Steps:**
- [ ] Create `.github/workflows/ci.yml`
- [ ] x86 job: `ubuntu-latest`, install deps, cmake build, run `test_correctness`
- [ ] ARM job: `ubuntu-latest` on ARM runner (or cross-compile + QEMU), build with
      SVE flags, run tests
- [ ] ARM SVE: implement `eval_sve` in `src/query_eval_sve.cpp` using
      `svand_u64_z` / `svbic_u64_z` — enough for CI to compile and pass correctness
- [ ] Badge in repo README

### D6: GitHub Pages site

**Why:** A polished web page is a better portfolio piece than a markdown README.
Readers can see charts, code snippets, and the narrative without cloning the repo.

**Narrative arc:** Problem → Measurement → Optimization → Result

**Sections:**
- [ ] The problem: audience segmentation at 500M users
- [ ] Why bitmaps: memory layout, cache lines, mechanical sympathy
- [ ] Scalar baseline and the auto-vectorization surprise
- [ ] AVX2: why naive SIMD lost, and how prefetching won
- [ ] Threading: write-allocate and why eval can't scale but popcount can
- [ ] Catching misleading benchmarks (the cold-baseline artifact)
- [ ] Comparative benchmarks: BitFilter vs CRoaring vs DuckDB vs SQLite
- [ ] Roofline model
- [ ] What this means for CPU architecture (the NVIDIA angle)

**Steps:**
- [ ] Set up `site/` directory with `index.html` + Pico.css (classless framework)
- [ ] Write content with embedded charts from D4 (SVG/PNG from matplotlib)
- [ ] Add interactive comparison chart via Chart.js or D3.js
- [ ] Deploy via GitHub Pages (Settings → Source → `site/` folder on `main`)

---

## Execution order

```
Phase 1 — Data collection (can parallelize)
  ├── D1: CRoaring benchmark
  ├── D2: DuckDB benchmark
  └── D3: SQLite benchmark

Phase 2 — Analysis
  └── D4: Roofline model (needs final numbers from Phase 1)

Phase 3 — Publishing (can parallelize)
  ├── D5: CI pipeline + ARM SVE stub implementation
  └── D6: GitHub Pages site (needs D4 chart + Phase 1 results)
```

---

## Decisions (locked)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| CRoaring | CMake FetchContent (source build, v4.x) | apt versions are often outdated and compiled with generic flags. Source build ensures CRoaring uses the same `-march=native` AVX2 optimizations. Shows dependency management skill. |
| DuckDB | C++ API (prebuilt `libduckdb.so` v1.5.1) | Apples-to-apples comparison — no Python interpreter overhead. Keeps the project pure C++. Prebuilt .so instead of amalgamation source: the 25 MB `duckdb.cpp` OOMs at `-O3` on machines with <12 GB RAM. |
| SQLite scale | 50M rows + extrapolation | 500M rows in a row-store tests SSD I/O, not database logic. ~40-60 GB disk, hours to load. Representative sub-sample is the industry-standard approach. Hook: "SQLite took 4 min for 50M; BitFilter took 11 ms for 500M." |
| GitHub Pages | Plain HTML + Pico.css (classless CSS) | Zero-dependency site mirrors the project philosophy — no bloat. Easier to embed interactive charts (Chart.js/D3) than fighting Jekyll's Liquid templates. |
| Roofline chart | Python matplotlib | Reproducible from script, export SVG/PNG for the site. |

---

## Definition of done

- [ ] CRoaring, DuckDB, SQLite numbers collected and recorded
- [ ] Roofline chart generated with all variants plotted
- [ ] CI green on both x86 and ARM
- [ ] GitHub Pages site live with full narrative + charts
- [ ] plan.md updated to reflect final Week 4 outcomes
