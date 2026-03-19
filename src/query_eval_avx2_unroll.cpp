#include "query_eval.hpp"
#include <immintrin.h>

// ── 2x unrolled AVX2 ────────────────────────────────────────────────────────
//
// Processes 2 YMM registers per iteration = 8 uint64_t = 512 bits.
// Doubles the number of independent loads visible to the out-of-order engine,
// improving memory-level parallelism at DRAM-bound scales.
void eval_avx2_unroll2(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words)
{
    size_t i = 0;
    constexpr size_t step = 8;  // 2 × 4 words per iteration

    for (; i + step <= n_words; i += step) {
        __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a     + i));
        __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a     + i + 4));
        __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b     + i));
        __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b     + i + 4));
        __m256i vc0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(not_c + i));
        __m256i vc1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(not_c + i + 4));

        __m256i tmp0 = _mm256_and_si256(va0, vb0);
        __m256i tmp1 = _mm256_and_si256(va1, vb1);
        __m256i vr0  = _mm256_andnot_si256(vc0, tmp0);
        __m256i vr1  = _mm256_andnot_si256(vc1, tmp1);

        _mm256_store_si256(reinterpret_cast<__m256i*>(result + i),     vr0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(result + i + 4), vr1);
    }

    // Scalar tail: 0–7 remaining words
    for (; i < n_words; ++i)
        result[i] = a[i] & b[i] & ~not_c[i];
}

// ── 4x unrolled AVX2 ────────────────────────────────────────────────────────
//
// Processes 4 YMM registers per iteration = 16 uint64_t = 1024 bits.
// Uses 12 YMM registers for loads + 4 for results = 16 total, which is the
// full AVX2 register file. Maximises instruction-level parallelism at the
// cost of a larger scalar tail (0–15 words).
void eval_avx2_unroll4(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words)
{
    size_t i = 0;
    constexpr size_t step = 16;  // 4 × 4 words per iteration

    for (; i + step <= n_words; i += step) {
        __m256i va0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a     + i));
        __m256i va1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a     + i + 4));
        __m256i va2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a     + i + 8));
        __m256i va3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(a     + i + 12));

        __m256i vb0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b     + i));
        __m256i vb1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b     + i + 4));
        __m256i vb2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b     + i + 8));
        __m256i vb3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(b     + i + 12));

        __m256i vc0 = _mm256_load_si256(reinterpret_cast<const __m256i*>(not_c + i));
        __m256i vc1 = _mm256_load_si256(reinterpret_cast<const __m256i*>(not_c + i + 4));
        __m256i vc2 = _mm256_load_si256(reinterpret_cast<const __m256i*>(not_c + i + 8));
        __m256i vc3 = _mm256_load_si256(reinterpret_cast<const __m256i*>(not_c + i + 12));

        __m256i tmp0 = _mm256_and_si256(va0, vb0);
        __m256i tmp1 = _mm256_and_si256(va1, vb1);
        __m256i tmp2 = _mm256_and_si256(va2, vb2);
        __m256i tmp3 = _mm256_and_si256(va3, vb3);

        __m256i vr0  = _mm256_andnot_si256(vc0, tmp0);
        __m256i vr1  = _mm256_andnot_si256(vc1, tmp1);
        __m256i vr2  = _mm256_andnot_si256(vc2, tmp2);
        __m256i vr3  = _mm256_andnot_si256(vc3, tmp3);

        _mm256_store_si256(reinterpret_cast<__m256i*>(result + i),      vr0);
        _mm256_store_si256(reinterpret_cast<__m256i*>(result + i + 4),  vr1);
        _mm256_store_si256(reinterpret_cast<__m256i*>(result + i + 8),  vr2);
        _mm256_store_si256(reinterpret_cast<__m256i*>(result + i + 12), vr3);
    }

    // Scalar tail: 0–15 remaining words
    for (; i < n_words; ++i)
        result[i] = a[i] & b[i] & ~not_c[i];
}
