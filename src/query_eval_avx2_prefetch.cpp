#include "query_eval.hpp"
#include <immintrin.h>

// AVX2 with software prefetching.
//
// Based on the 1-register-per-iteration AVX2 loop (the best-performing SIMD
// variant in benchmarks). Adds _mm_prefetch calls to request data into L1
// before we need it, hiding DRAM latency by overlapping fetch with compute.
//
// Prefetch distance: PREFETCH_DIST words ahead (each word = 8 bytes).
// At 16 words = 128 bytes = 2 cache lines ahead. This is a reasonable
// starting point for DDR4 latency (~60-80ns). The optimal distance depends
// on memory latency and loop throughput — tuned empirically since perf
// hardware counters are unavailable on WSL2.
//
// _MM_HINT_T0: prefetch into L1 (and all higher levels). We use T0 because
// the data will be consumed immediately in the next few iterations.
void eval_avx2_prefetch(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words)
{
    size_t i = 0;
    constexpr size_t simd_width    = 4;   // 4 × uint64_t = 256 bits
    constexpr size_t prefetch_dist = 16;  // 16 words = 128 bytes ahead

    for (; i + simd_width <= n_words; i += simd_width) {
        // Prefetch future cache lines for all four arrays
        if (i + prefetch_dist < n_words) {
            _mm_prefetch(reinterpret_cast<const char*>(a     + i + prefetch_dist), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(b     + i + prefetch_dist), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(not_c + i + prefetch_dist), _MM_HINT_T0);
            _mm_prefetch(reinterpret_cast<const char*>(result+ i + prefetch_dist), _MM_HINT_T0);
        }

        __m256i va  = _mm256_load_si256(reinterpret_cast<const __m256i*>(a     + i));
        __m256i vb  = _mm256_load_si256(reinterpret_cast<const __m256i*>(b     + i));
        __m256i vc  = _mm256_load_si256(reinterpret_cast<const __m256i*>(not_c + i));

        __m256i tmp = _mm256_and_si256(va, vb);
        __m256i vr  = _mm256_andnot_si256(vc, tmp);

        _mm256_store_si256(reinterpret_cast<__m256i*>(result + i), vr);
    }

    // Scalar tail
    for (; i < n_words; ++i)
        result[i] = a[i] & b[i] & ~not_c[i];
}
