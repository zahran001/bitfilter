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

// ── Test 5: eval_avx2 matches eval_scalar (bit-exact) ────────────────────────
//
// The scalar path is the correctness reference. The AVX2 path must produce
// identical output for every word — no exceptions, no tolerance.
//
// We test three word counts to exercise different code paths in eval_avx2:
//   8192  — multiple of 4: pure SIMD, zero tail words
//   8195  — 8192 + 3 remainder: exercises the scalar tail loop for all 3
//           possible remainder values (n_words % 4 == 3)
//   1     — single word: no SIMD iterations at all, pure tail
//
// Each test generates three input bitmaps (a, b, not_c) with independent seeds,
// runs both eval functions, and asserts memcmp == 0 on the result buffers.
// If this fails, the bug is definitively in eval_avx2 — not in the reference.

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

class EvalAvx2Correctness : public ::testing::TestWithParam<size_t> {};

TEST_P(EvalAvx2Correctness, MatchesScalar) {
    const size_t n_words = GetParam();

    // Allocate inputs + two result buffers (all 64-byte aligned)
    AlignedBuffer a            = make_aligned_bitmap(n_words);
    AlignedBuffer b            = make_aligned_bitmap(n_words);
    AlignedBuffer not_c        = make_aligned_bitmap(n_words);
    AlignedBuffer result_scalar = make_aligned_bitmap(n_words);
    AlignedBuffer result_avx2   = make_aligned_bitmap(n_words);

    // Fill inputs with independent random data
    fill_random_bitmap(a.get(),     n_words, 0xAAAA'AAAA'AAAA'AAAAULL);
    fill_random_bitmap(b.get(),     n_words, 0xBBBB'BBBB'BBBB'BBBBULL);
    fill_random_bitmap(not_c.get(), n_words, 0xCCCC'CCCC'CCCC'CCCCULL);

    // Zero both result buffers so any unevaluated word is visible
    std::memset(result_scalar.get(), 0, n_words * sizeof(uint64_t));
    std::memset(result_avx2.get(),   0, n_words * sizeof(uint64_t));

    // Run both paths
    eval_scalar(a.get(), b.get(), not_c.get(), result_scalar.get(), n_words);
    eval_avx2  (a.get(), b.get(), not_c.get(), result_avx2.get(),   n_words);

    // Bit-exact comparison
    EXPECT_EQ(std::memcmp(result_scalar.get(), result_avx2.get(),
                          n_words * sizeof(uint64_t)), 0)
        << "eval_avx2 output differs from eval_scalar at n_words=" << n_words;
}

INSTANTIATE_TEST_SUITE_P(
    WordCounts,
    EvalAvx2Correctness,
    ::testing::Values(
        8192,   // multiple of 4: pure SIMD, no tail
        8195,   // remainder 3: exercises all tail positions
        1       // single word: pure tail, no SIMD iterations
    ),
    [](const ::testing::TestParamInfo<size_t>& info) {
        return "n_words_" + std::to_string(info.param);
    }
);