#include <gtest/gtest.h>
#include "segment_store.hpp"
#include "query_eval.hpp"
#include "aligned_alloc.hpp"
#include "datagen.hpp"
#include <cstring>
#include <cstdint>
#include <random>

// ── Test 1: zero-initialisation ───────────────────────────────────────────────
//
// After add_segment, every bit must be zero.
// This catches the bug where aligned_alloc returns uninitialised memory and
// the caller forgets to memset — a real failure mode that produces phantom
// users in every query result.
TEST(SegmentStore, ZeroInitAfterAddSegment) {
    SegmentStore store(1'000'000);
    store.add_segment("a");

    const uint64_t* bm = store.get_bitmap("a");
    uint64_t total = popcount(bm, store.n_words());

    EXPECT_EQ(total, 0u)
        << "Bitmap must be zero-initialised. Got " << total << " set bits.";
}

// ── Test 2: set_bit / get_bit round-trip ─────────────────────────────────────
//
// Verify that set_bit writes to the correct word and bit position, and that
// get_bit reads from the same position.
//
// The bit layout contract: user ID u → word u/64, bit u%64 (LSB = user 0).
// If set_bit and get_bit use different conventions (e.g. one uses MSB order),
// they will appear to work in isolation but disagree with each other and with
// the eval functions.
//
// We test three user IDs:
//   0        — first bit of first word (boundary)
//   63       — last bit of first word (boundary)
//   64       — first bit of second word (word-crossing)
//   999999   — near the end of the population
TEST(SegmentStore, SetBitGetBitRoundTrip) {
    SegmentStore store(1'000'000);
    store.add_segment("a");
    uint64_t* bm = store.get_bitmap_mut("a");

    const size_t test_ids[] = {0, 63, 64, 999'999};

    for (size_t uid : test_ids) {
        // Verify not set before we set it
        EXPECT_FALSE(get_bit(bm, uid))
            << "uid=" << uid << " should be 0 before set_bit";

        set_bit(bm, uid);

        EXPECT_TRUE(get_bit(bm, uid))
            << "uid=" << uid << " should be 1 after set_bit";
    }

    // Verify that setting uid N did not contaminate uid N+1 or N-1
    // (catches off-by-one in bit shift)
    EXPECT_FALSE(get_bit(bm, 1))   << "uid=1 should not be set (only 0 was set)";
    EXPECT_FALSE(get_bit(bm, 62))  << "uid=62 should not be set (only 63 was set)";
    EXPECT_FALSE(get_bit(bm, 65))  << "uid=65 should not be set (only 64 was set)";
}

// ── Test 3: popcount matches manual count ─────────────────────────────────────
//
// Set a known number of bits via DataGen, then verify popcount returns
// the expected value within statistical tolerance.
//
// Why a tolerance rather than an exact count?
// std::bernoulli_distribution produces counts that are binomially distributed.
// At density=0.5, n=1'000'000, the standard deviation is sqrt(n*0.5*0.5)=500.
// We use ±5 sigma (2500) as the tolerance — this will virtually never fail
// on a correct implementation, but would catch popcount returning 0 or n_users.
//
// This test also validates that padding bits (positions [n_users..n_words*64))
// are never set — if they were, popcount would return more than n_users.
TEST(SegmentStore, PopcountMatchesDensity) {
    const size_t n_users  = 1'000'000;
    const double density  = 0.5;
    const uint64_t seed   = 42;

    SegmentStore store(n_users);
    store.add_segment("a");
    DataGen::populate(store, "a", density, seed);

    const uint64_t* bm = store.get_bitmap("a");
    uint64_t count = popcount(bm, store.n_words());

    // Sanity: count must not exceed n_users (would mean padding bits are set)
    EXPECT_LE(count, n_users)
        << "popcount exceeded n_users — padding bits are set!";

    // Statistical: count should be near density * n_users
    double expected   = density * static_cast<double>(n_users);
    double sigma      = std::sqrt(expected * (1.0 - density));
    double tolerance  = 5.0 * sigma;  // 5-sigma: P(false failure) < 1 in 3.5M

    EXPECT_NEAR(static_cast<double>(count), expected, tolerance)
        << "popcount=" << count << " is more than 5 sigma from expected="
        << expected << " (sigma=" << sigma << ")";
}

// ── Test 4: duplicate segment name is rejected ────────────────────────────────
//
// add_segment must throw if the name already exists. Without this guard,
// a second add_segment silently discards the first bitmap (because
// unordered_map::emplace is a no-op on collision) and the caller ends up
// holding a stale pointer — a dangling-pointer bug that only surfaces on the
// next query.
TEST(SegmentStore, DuplicateSegmentThrows) {
    SegmentStore store(1'000'000);
    store.add_segment("dup");

    EXPECT_THROW(store.add_segment("dup"), std::invalid_argument)
        << "Adding a duplicate segment name must throw std::invalid_argument";
}

// ── Test 5: all AVX2 variants match eval_scalar (bit-exact) ──────────────────
//
// The scalar path is the correctness reference. Every AVX2 variant must produce
// identical output for every word — no exceptions, no tolerance.
//
// We test three word counts to exercise different code paths:
//   8192  — multiple of 16: pure SIMD for all variants, zero tail words
//   8195  — remainder 3: exercises scalar tail in all variants
//   1     — single word: pure tail, no SIMD iterations at all
//
// Each test generates three input bitmaps (a, b, not_c) with independent seeds,
// runs eval_scalar and one AVX2 variant, and asserts memcmp == 0.

// Helper: fill an aligned bitmap with random bits (per-bit bernoulli, density 0.5).
static void fill_random_bitmap(uint64_t* words, size_t n_words, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution dist(0.5);
    for (size_t w = 0; w < n_words; ++w) {
        uint64_t word = 0;
        for (int b = 0; b < 64; ++b)
            if (dist(rng)) word |= (UINT64_C(1) << b);
        words[w] = word;
    }
}

// Function pointer type for all eval variants.
using EvalFn = void(*)(
    const uint64_t* __restrict__,
    const uint64_t* __restrict__,
    const uint64_t* __restrict__,
          uint64_t* __restrict__,
    size_t);

struct EvalVariant {
    const char* name;
    EvalFn      fn;
};

// All AVX2 variants to test against the scalar reference.
static const EvalVariant kEvalVariants[] = {
    {"avx2",          eval_avx2},
    {"avx2_unroll2",  eval_avx2_unroll2},
    {"avx2_unroll4",  eval_avx2_unroll4},
    {"avx2_prefetch", eval_avx2_prefetch},
};

struct EvalTestParam {
    size_t       n_words;
    EvalVariant  variant;
};

class EvalCorrectness : public ::testing::TestWithParam<EvalTestParam> {};

TEST_P(EvalCorrectness, MatchesScalar) {
    const auto& [n_words, variant] = GetParam();

    AlignedBuffer a              = make_aligned_bitmap(n_words);
    AlignedBuffer b              = make_aligned_bitmap(n_words);
    AlignedBuffer not_c          = make_aligned_bitmap(n_words);
    AlignedBuffer result_scalar  = make_aligned_bitmap(n_words);
    AlignedBuffer result_variant = make_aligned_bitmap(n_words);

    fill_random_bitmap(a.get(),     n_words, 0xAAAA'AAAA'AAAA'AAAAULL);
    fill_random_bitmap(b.get(),     n_words, 0xBBBB'BBBB'BBBB'BBBBULL);
    fill_random_bitmap(not_c.get(), n_words, 0xCCCC'CCCC'CCCC'CCCCULL);

    std::memset(result_scalar.get(),  0, n_words * sizeof(uint64_t));
    std::memset(result_variant.get(), 0, n_words * sizeof(uint64_t));

    eval_scalar(a.get(), b.get(), not_c.get(), result_scalar.get(),  n_words);
    variant.fn (a.get(), b.get(), not_c.get(), result_variant.get(), n_words);

    EXPECT_EQ(std::memcmp(result_scalar.get(), result_variant.get(),
                          n_words * sizeof(uint64_t)), 0)
        << variant.name << " output differs from eval_scalar at n_words=" << n_words;
}

// Generate the cross-product: 3 variants × 3 word counts = 9 test cases.
static std::vector<EvalTestParam> MakeEvalParams() {
    std::vector<EvalTestParam> params;
    for (size_t n : {8192, 8195, 1})
        for (const auto& v : kEvalVariants)
            params.push_back({n, v});
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    AllVariants,
    EvalCorrectness,
    ::testing::ValuesIn(MakeEvalParams()),
    [](const ::testing::TestParamInfo<EvalTestParam>& info) {
        return std::string(info.param.variant.name) + "_n" +
               std::to_string(info.param.n_words);
    }
);

// ── Test 6: eval_avx2_prefetch_mt matches single-threaded eval (bit-exact) ──
//
// The MT wrapper must produce identical output to the single-threaded kernel
// for every thread count and word count combination.
//
// Thread counts: 1 (fallback path), 2, 4, 12 (all logical cores)
// Word counts:   8192 (aligned), 8195 (remainder), 1 (degenerate)

struct MtEvalTestParam {
    size_t   n_words;
    unsigned n_threads;
};

class MtEvalCorrectness : public ::testing::TestWithParam<MtEvalTestParam> {};

TEST_P(MtEvalCorrectness, MatchesSingleThreaded) {
    const auto& [n_words, n_threads] = GetParam();

    AlignedBuffer a         = make_aligned_bitmap(n_words);
    AlignedBuffer b         = make_aligned_bitmap(n_words);
    AlignedBuffer not_c     = make_aligned_bitmap(n_words);
    AlignedBuffer result_st = make_aligned_bitmap(n_words);
    AlignedBuffer result_mt = make_aligned_bitmap(n_words);

    fill_random_bitmap(a.get(),     n_words, 0xAAAA'AAAA'AAAA'AAAAULL);
    fill_random_bitmap(b.get(),     n_words, 0xBBBB'BBBB'BBBB'BBBBULL);
    fill_random_bitmap(not_c.get(), n_words, 0xCCCC'CCCC'CCCC'CCCCULL);

    std::memset(result_st.get(), 0, n_words * sizeof(uint64_t));
    std::memset(result_mt.get(), 0, n_words * sizeof(uint64_t));

    eval_avx2_prefetch(a.get(), b.get(), not_c.get(), result_st.get(), n_words);
    eval_avx2_prefetch_mt(a.get(), b.get(), not_c.get(), result_mt.get(),
                          n_words, n_threads);

    EXPECT_EQ(std::memcmp(result_st.get(), result_mt.get(),
                          n_words * sizeof(uint64_t)), 0)
        << "MT output differs from single-threaded at n_words=" << n_words
        << ", n_threads=" << n_threads;
}

static std::vector<MtEvalTestParam> MakeMtEvalParams() {
    std::vector<MtEvalTestParam> params;
    for (size_t n : {8192, 8195, 1})
        for (unsigned t : {1u, 2u, 4u, 12u})
            params.push_back({n, t});
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    MtVariants,
    MtEvalCorrectness,
    ::testing::ValuesIn(MakeMtEvalParams()),
    [](const ::testing::TestParamInfo<MtEvalTestParam>& info) {
        return "n" + std::to_string(info.param.n_words) +
               "_t" + std::to_string(info.param.n_threads);
    }
);

// ── Test 7: popcount_mt matches single-threaded popcount (exact) ────────────

struct MtPopcountTestParam {
    size_t   n_words;
    unsigned n_threads;
};

class MtPopcountCorrectness : public ::testing::TestWithParam<MtPopcountTestParam> {};

TEST_P(MtPopcountCorrectness, MatchesSingleThreaded) {
    const auto& [n_words, n_threads] = GetParam();

    AlignedBuffer bm = make_aligned_bitmap(n_words);
    fill_random_bitmap(bm.get(), n_words, 0xDDDD'DDDD'DDDD'DDDDULL);

    uint64_t expected = popcount(bm.get(), n_words);
    uint64_t actual   = popcount_mt(bm.get(), n_words, n_threads);

    EXPECT_EQ(actual, expected)
        << "popcount_mt differs at n_words=" << n_words
        << ", n_threads=" << n_threads
        << " (expected=" << expected << ", got=" << actual << ")";
}

static std::vector<MtPopcountTestParam> MakeMtPopcountParams() {
    std::vector<MtPopcountTestParam> params;
    for (size_t n : {8192, 8195, 1})
        for (unsigned t : {1u, 2u, 4u, 12u})
            params.push_back({n, t});
    return params;
}

INSTANTIATE_TEST_SUITE_P(
    MtPopcountVariants,
    MtPopcountCorrectness,
    ::testing::ValuesIn(MakeMtPopcountParams()),
    [](const ::testing::TestParamInfo<MtPopcountTestParam>& info) {
        return "n" + std::to_string(info.param.n_words) +
               "_t" + std::to_string(info.param.n_threads);
    }
);