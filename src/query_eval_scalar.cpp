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
