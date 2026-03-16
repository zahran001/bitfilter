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
