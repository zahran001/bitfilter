// ARM SVE path for NVIDIA Grace (ARMv9) and Ampere Altra (ARMv8+SVE).
// Uses scalable vector length — the same binary works across all SVE
// implementations regardless of hardware vector width (128-bit to 2048-bit).
//
// Not compiled on x86 — guarded by __ARM_FEATURE_SVE.

#if defined(__ARM_FEATURE_SVE)
#include "query_eval.hpp"
#include <arm_sve.h>

void eval_sve(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words)
{
    size_t i = 0;
    while (i < n_words) {
        // svwhilelt generates a predicate mask for the remaining elements,
        // handling the tail automatically — no scalar cleanup needed.
        svbool_t pg = svwhilelt_b64(i, n_words);

        svuint64_t va = svld1_u64(pg, a + i);
        svuint64_t vb = svld1_u64(pg, b + i);
        svuint64_t vc = svld1_u64(pg, not_c + i);

        // a & b & ~c  →  svbic(a & b, c)  since BIC(x, y) = x AND NOT y
        svuint64_t tmp = svand_u64_z(pg, va, vb);
        svuint64_t vr  = svbic_u64_z(pg, tmp, vc);

        svst1_u64(pg, result + i, vr);

        i += svcntd();  // advance by hardware vector length in uint64_t elements
    }
}

#endif // __ARM_FEATURE_SVE
