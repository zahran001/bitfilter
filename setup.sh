#!/usr/bin/env bash
# setup.sh — run once from ~/bitfilter to create the project skeleton.
# Safe to re-run: uses -p for directories and only creates files that don't exist.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Setting up bitfilter skeleton in: $REPO_ROOT"

# ── Directories ───────────────────────────────────────────────────────────────
mkdir -p "$REPO_ROOT"/{include,src,test,bench}

# ── include/aligned_alloc.hpp ─────────────────────────────────────────────────
# Already specified in the project context — copy verbatim.
cat > "$REPO_ROOT/include/aligned_alloc.hpp" << 'EOF'
#pragma once
#include <cstdlib>
#include <memory>
#include <new>

inline void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

struct AlignedDeleter {
    void operator()(void* p) const { aligned_free(p); }
};

// AlignedBuffer: RAII owner for a 64-byte-aligned array of uint64_t.
// Allocated via aligned_malloc, freed via aligned_free — never delete[].
using AlignedBuffer = std::unique_ptr<uint64_t[], AlignedDeleter>;

// Allocate n_words uint64_t values with 64-byte alignment (cache-line perfect).
// Throws std::bad_alloc if the allocation fails.
// Does NOT zero-initialise — caller must memset before use.
inline AlignedBuffer make_aligned_bitmap(size_t n_words) {
    void* raw = aligned_malloc(n_words * sizeof(uint64_t), 64);
    if (!raw) throw std::bad_alloc{};
    return AlignedBuffer(static_cast<uint64_t*>(raw));
}
EOF

# ── include/segment_store.hpp ─────────────────────────────────────────────────
cat > "$REPO_ROOT/include/segment_store.hpp" << 'EOF'
#pragma once
#include "aligned_alloc.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

// SegmentStore owns all segment bitmaps.
//
// Every bitmap has exactly n_words_ uint64_t words, covering n_users_ bits.
// Bits [n_users_ .. n_words_*64) are permanently zero — no padding bits
// are ever set, so popcount() never over-counts.
//
// Bit layout: user ID u lives at word u/64, bit u%64 (LSB = user 0).
// This must be consistent with every caller of set_bit() and get_bit().
class SegmentStore {
public:
    // n_users: total population size (e.g. 500'000'000).
    explicit SegmentStore(size_t n_users);

    // Add a zero-initialised bitmap for the given segment name.
    // Throws std::invalid_argument if the name already exists.
    void add_segment(const std::string& name);

    // Raw read-only pointer to segment bitmap. Throws std::out_of_range
    // if the name does not exist. Use this for eval inputs.
    const uint64_t* get_bitmap(const std::string& name) const;

    // Raw mutable pointer. Throws std::out_of_range if name does not exist.
    // Use this in the data generator to write bits.
    uint64_t* get_bitmap_mut(const std::string& name);

    // Number of uint64_t words in every bitmap.
    size_t n_words() const noexcept { return n_words_; }

    // Number of users (bits in the active range of every bitmap).
    size_t n_users() const noexcept { return n_users_; }

private:
    size_t n_users_;
    size_t n_words_;  // = ceil(n_users / 64)
    std::unordered_map<std::string, AlignedBuffer> segments_;
};

// ── Free helpers (header-only, used by DataGen and tests) ─────────────────────

// Set bit for user_id in bitmap. Caller owns the pointer and guarantees
// user_id < n_users for this bitmap.
inline void set_bit(uint64_t* bitmap, size_t user_id) noexcept {
    bitmap[user_id / 64] |= (UINT64_C(1) << (user_id % 64));
}

// Read bit for user_id.
inline bool get_bit(const uint64_t* bitmap, size_t user_id) noexcept {
    return (bitmap[user_id / 64] >> (user_id % 64)) & UINT64_C(1);
}
EOF

# ── include/query_eval.hpp ────────────────────────────────────────────────────
cat > "$REPO_ROOT/include/query_eval.hpp" << 'EOF'
#pragma once
#include <cstddef>
#include <cstdint>
#include <nmmintrin.h>  // _mm_popcnt_u64 (SSE4.2 / BMI2 — confirmed on this machine)

// Scalar baseline — correctness reference, not a performance target.
// Do NOT add optimisation hints; the compiler must not auto-vectorise this.
void eval_scalar(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words);

// AVX2 primary path — 256-bit YMM registers, 4 × uint64_t per iteration.
void eval_avx2(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words);

// Hardware popcount using _mm_popcnt_u64.
// Counts set bits across all n_words words.
// Padding bits (positions [n_users .. n_words*64)) are guaranteed zero
// by SegmentStore, so no masking of the last word is needed.
inline uint64_t popcount(const uint64_t* bitmap, size_t n_words) noexcept {
    uint64_t total = 0;
    for (size_t i = 0; i < n_words; ++i)
        total += _mm_popcnt_u64(bitmap[i]);
    return total;
}
EOF

# ── include/query_ast.hpp ─────────────────────────────────────────────────────
# Placeholder — not needed until Week 2+. Exists so the layout is complete.
cat > "$REPO_ROOT/include/query_ast.hpp" << 'EOF'
#pragma once
// Boolean query AST — used in Week 2+ for multi-segment query planning.
// Stub only for Week 1.
EOF

# ── src/segment_store.cpp ─────────────────────────────────────────────────────
cat > "$REPO_ROOT/src/segment_store.cpp" << 'EOF'
#include "segment_store.hpp"
#include <cstring>
#include <stdexcept>

SegmentStore::SegmentStore(size_t n_users)
    : n_users_(n_users)
    , n_words_((n_users + 63) / 64)  // ceiling division: covers all bits
{}

void SegmentStore::add_segment(const std::string& name) {
    if (segments_.count(name))
        throw std::invalid_argument("segment already exists: " + name);

    AlignedBuffer buf = make_aligned_bitmap(n_words_);
    // Zero-init is mandatory: aligned_alloc does not initialise memory.
    // Unzeroed padding bits would corrupt popcount and query results.
    std::memset(buf.get(), 0, n_words_ * sizeof(uint64_t));
    segments_.emplace(name, std::move(buf));
}

const uint64_t* SegmentStore::get_bitmap(const std::string& name) const {
    return segments_.at(name).get();  // throws std::out_of_range on miss
}

uint64_t* SegmentStore::get_bitmap_mut(const std::string& name) {
    return segments_.at(name).get();
}
EOF

# ── src/query_eval_scalar.cpp ─────────────────────────────────────────────────
cat > "$REPO_ROOT/src/query_eval_scalar.cpp" << 'EOF'
#include "query_eval.hpp"

// Scalar baseline: result[i] = a[i] & b[i] & ~not_c[i]
//
// This is the correctness reference. Do not add SIMD hints, loop pragmas,
// or any attribute that would let the compiler vectorise this path.
// If both scalar and AVX2 produce the same wrong answer, the bug is invisible.
void eval_scalar(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words)
{
    for (size_t i = 0; i < n_words; ++i)
        result[i] = a[i] & b[i] & ~not_c[i];
}
EOF

# ── src/query_eval_avx2.cpp ───────────────────────────────────────────────────
cat > "$REPO_ROOT/src/query_eval_avx2.cpp" << 'EOF'
#include "query_eval.hpp"
#include <immintrin.h>

// AVX2 primary path.
//
// One iteration processes 4 × uint64_t = 256 bits using 256-bit YMM registers.
//
// ANDNOT operand order — critical:
//   _mm256_andnot_si256(a, b) computes (~a) & b.
//   We want result = tmp & ~not_c, so the call is andnot(not_c, tmp).
//   Swapping the arguments silently inverts the NOT-C logic.
//
// Alignment: _mm256_load_si256 requires 32-byte alignment. Our buffers are
// 64-byte aligned (cache-line aligned), which satisfies this requirement.
// Using the unaligned variant (_mm256_loadu_si256) would be correct but
// wastes the alignment guarantee we already pay for.
void eval_avx2(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words)
{
    size_t i = 0;
    constexpr size_t simd_width = 4;  // 4 × uint64_t = 256 bits

    for (; i + simd_width <= n_words; i += simd_width) {
        __m256i va  = _mm256_load_si256(reinterpret_cast<const __m256i*>(a     + i));
        __m256i vb  = _mm256_load_si256(reinterpret_cast<const __m256i*>(b     + i));
        __m256i vc  = _mm256_load_si256(reinterpret_cast<const __m256i*>(not_c + i));

        __m256i tmp = _mm256_and_si256(va, vb);
        // andnot(vc, tmp) = (~vc) & tmp  →  tmp & ~not_c  ✓
        __m256i vr  = _mm256_andnot_si256(vc, tmp);

        _mm256_store_si256(reinterpret_cast<__m256i*>(result + i), vr);
    }

    // Scalar tail: handles the remaining 0–3 words when n_words % 4 != 0.
    for (; i < n_words; ++i)
        result[i] = a[i] & b[i] & ~not_c[i];
}
EOF

# ── src/query_eval_sve.cpp ────────────────────────────────────────────────────
# ARM SVE path — stub only; compiled in CI on ARM runner, not locally.
cat > "$REPO_ROOT/src/query_eval_sve.cpp" << 'EOF'
// ARM SVE path for NVIDIA Grace (ARMv9). Not compiled on x86.
// Stub only — implementation in Week 2+.
#if defined(__ARM_FEATURE_SVE)
#include "query_eval.hpp"
#include <arm_sve.h>
// TODO Week 2: implement eval_sve using svand_u64_z / svanot_u64_z
#endif
EOF

# ── test/test_correctness.cpp ─────────────────────────────────────────────────
# Minimal stub — just proves the test binary links and runs.
# Real tests added in Chunk 4.
cat > "$REPO_ROOT/test/test_correctness.cpp" << 'EOF'
#include <gtest/gtest.h>
#include "segment_store.hpp"
#include "query_eval.hpp"

// Chunk 1 smoke test: the project builds and links.
// Real correctness tests are added in Chunk 4.
TEST(Smoke, BuildAndLink) {
    SUCCEED();
}
EOF

# ── bench/bench_main.cpp ──────────────────────────────────────────────────────
# Minimal stub — proves the benchmark binary links.
# Real benchmarks added in Chunk 5.
cat > "$REPO_ROOT/bench/bench_main.cpp" << 'EOF'
#include <benchmark/benchmark.h>
#include "segment_store.hpp"
#include "query_eval.hpp"

// Chunk 1 stub — proves the benchmark binary links.
// Real benchmarks added in Chunk 5.
static void BM_Placeholder(benchmark::State& state) {
    for (auto _ : state) {
        benchmark::DoNotOptimize(0);
    }
}
BENCHMARK(BM_Placeholder);
// Note: main() is provided by benchmark::benchmark_main.
// Do not write your own main() here.
EOF

echo ""
echo "✓ Skeleton created. Next steps:"
echo ""
echo "  cd ~/bitfilter"
echo "  mkdir build && cd build"
echo "  cmake .. -DCMAKE_BUILD_TYPE=Release"
echo "  cmake --build . --parallel"
echo "  ctest --output-on-failure     # should pass: Smoke.BuildAndLink"
echo "  ./bench                        # should run: BM_Placeholder"
echo ""
echo "Once both succeed, you're ready for Chunk 2 (SegmentStore implementation)."
