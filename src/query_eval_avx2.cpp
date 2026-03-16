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
