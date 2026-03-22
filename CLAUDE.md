# BitFilter — project invariants for Claude Code

## What this project is
A SIMD-accelerated audience segmentation engine in C++20.
Goal: answer boolean bitmap queries (A AND B AND NOT C) over 500M users
at memory-bandwidth speed. Portfolio project for NVIDIA DevTech CPU role.

---

## Hardware — development machine
- Intel 12th/13th Gen hybrid: 2 P-cores (HT) + 8 E-cores = 12 logical processors
- AVX2 ✅  BMI2 ✅  POPCNT ✅
- AVX-512 ❌ — permanently disabled by Intel on hybrid chips, do NOT use it
- Cache line: 64 bytes — this drives all alignment decisions in this codebase

---

## Non-negotiable code rules

**SIMD path: AVX2 only (256-bit YMM registers)**
Never write AVX-512 intrinsics (`_mm512_*`, `__m512i`). The machine will crash.
Use `_mm256_*` and `__m256i` exclusively for the x86 path.

**Alignment: always 64 bytes**
All bitmap allocations must use `make_aligned_bitmap()` from `include/aligned_alloc.hpp`.
Never use `new`, `malloc`, or `std::aligned_alloc` directly for bitmap data.

**Popcount: hardware instruction only**
Use `_mm_popcnt_u64(data[i])` in `Bitmap::popcount()`.
Never use naive bit-counting loops or `__builtin_popcountll` in production paths.

**Scalar baseline must always exist**
`eval_scalar` in `src/query_eval_scalar.cpp` is the correctness reference.
Every optimization must be validated against it via `memcmp == 0` before merging.

**Benchmark link flags**
Always: `-lbenchmark -lbenchmark_main -lpthread`
`-lbenchmark` alone causes a linker failure even when headers are found.

---

## Environment

**Shell context: Claude Code's shell runs inside Git Bash on Windows, NOT inside WSL2.**
All commands that touch the Linux environment must be prefixed with `wsl -d Ubuntu -- bash -c "..."`.
Never assume `dpkg`, `g++`, `/proc`, or any Linux path is directly accessible from the shell — it isn't.

- OS: WSL2 Ubuntu 24.04.4 LTS
- Compiler: g++ 13.3.0
- Build system: CMake 3.28.3 + Ninja
- Project root: `/home/zahran1/projects/bitfilter`
- CMake flags (x86): `-mavx2 -mbmi2 -mpopcnt -O3 -march=native`
- CMake flags (ARM): `-march=armv9-a+sve2 -O3`

**perf hardware counters are blocked on WSL2.**
Do not attempt to collect `cycles`, `cache-misses`, or IPC locally.
Week 3 profiling runs on Akamai Cloud (Ubuntu 24.04 bare metal).

---

## Current phase
Week 1 — Foundation and correctness.
Nothing is optimized yet. Write correct code first, fast code second.

## Project plan
Full README, 4-week plan, and all decisions:
Refer to the plan.md file at the root.

**Any change to scope, architecture, tooling, or decisions made during
implementation must be reflected in plan.md immediately. plan.md is the
single source of truth for this project. It does not drift from reality.**

## Documentation
The `docs/` folder at the project root contains supplementary project documentation.
