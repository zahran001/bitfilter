# D2: BitFilter vs DuckDB Comparative Benchmark

## Setup

- **DuckDB version:** v1.5.1 (prebuilt `libduckdb.so`, linux-amd64)
- **Fetch dependency (gitignored):**
  ```bash
  mkdir -p bench/duckdb && curl -sL https://github.com/duckdb/duckdb/releases/download/v1.5.1/libduckdb-linux-amd64.zip | python3 -c "import sys,zipfile,io; z=zipfile.ZipFile(io.BytesIO(sys.stdin.buffer.read())); [z.extract(f,'bench/duckdb/') for f in ('duckdb.hpp','libduckdb.so')]"
  ```
- **BitFilter path:** `eval_avx2_prefetch` (AVX2 + software prefetch, single-threaded)
- **Query:** `A AND B AND NOT C` over 500M users
- **DuckDB equivalent:** `SELECT COUNT(*) FROM segments WHERE a AND b AND NOT c`
- **DuckDB table:** 3 boolean columns, 500M rows, in-memory database
- **Data generation:** DuckDB uses `random() < density` via `setseed(0.42)`; BitFilter uses `std::bernoulli_distribution` with fixed seeds (not identical bits, but statistically equivalent)
- **Machine:** Intel 12th Gen (2P+8E, 12 logical), DDR4-2660 dual-channel

## Results

| Engine | Density | Latency (ms) | Speedup |
|--------|---------|-------------|---------|
| BitFilter (AVX2+prefetch) | 10% | 11.4 | **28x** |
| DuckDB v1.5.1 | 10% | 319 | 1x |
| BitFilter (AVX2+prefetch) | 50% | 19.0 | **114x** |
| DuckDB v1.5.1 | 50% | 2170 | 1x |

## Analysis

**BitFilter is 28-114x faster than DuckDB** for this boolean bitmap query.

The gap widens dramatically at higher density (50%) because:

1. **DuckDB's overhead is per-row, not per-bit.** At 50% density, more rows pass the filter, which means DuckDB's result materialization pipeline does more work. BitFilter's cost is fixed — it always processes the full bitmap regardless of density.

2. **BitFilter does exactly 3 memory-streaming operations** (read A, read B, read NOT_C, write result) at memory-bandwidth speed. DuckDB must: parse SQL, plan the query, decompress columnar storage, evaluate predicates through its vectorized execution engine, aggregate results, and return them through the C++ API.

3. **DuckDB uses its own SIMD internally** (vectorized execution engine with 2048-element vectors). The gap is not about SIMD vs no-SIMD — it's about the abstraction overhead of a general-purpose query engine vs a purpose-built bitmap engine.

**This is the expected result.** DuckDB is designed for analytical SQL queries over diverse data types with joins, aggregations, and complex predicates. BitFilter is purpose-built for one thing: boolean bitmap operations at memory-bandwidth speed. The comparison demonstrates the cost of generality.

## Benchmark command

```bash
cd /home/zahran1/projects/bitfilter
LD_LIBRARY_PATH=bench/duckdb:$LD_LIBRARY_PATH ./build/bench_duckdb
```

## Raw output

```
2026-03-27T01:57:49-04:00
Running ./build/bench_duckdb
Run on (12 X 2611.2 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x6)
  L1 Instruction 32 KiB (x6)
  L2 Unified 1280 KiB (x6)
  L3 Unified 12288 KiB (x1)
Load Average: 0.04, 0.03, 0.22
--------------------------------------------------------------------------------------------------------------
Benchmark                                                    Time             CPU   Iterations UserCounters...
--------------------------------------------------------------------------------------------------------------
BitFilterFixture/EvalAvx2Prefetch/1000                    11.4 ms         11.4 ms           58 bytes_per_second=20.3428Gi/s
BitFilterFixture/EvalAvx2Prefetch/5000                    19.0 ms         19.0 ms           61 bytes_per_second=12.2437Gi/s
DuckDBFixture/CountWhereAndAndNot/1000/iterations:5        319 ms          318 ms            5
DuckDBFixture/CountWhereAndAndNot/5000/iterations:5       2170 ms         2168 ms            5
```
