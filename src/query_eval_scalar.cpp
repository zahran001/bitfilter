#include "query_eval.hpp"

// Scalar baseline: result[i] = a[i] & b[i] & ~not_c[i]
//
// This is the correctness reference — it must execute genuine scalar code.
// Without no-tree-vectorize, GCC -O3 -march=native auto-vectorizes this
// loop to AVX2, which defeats its purpose: if both paths use the same SIMD
// instructions, an ANDNOT operand swap would produce identical wrong output
// and the memcmp correctness check would pass on a broken implementation.
__attribute__((optimize("no-tree-vectorize")))
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
