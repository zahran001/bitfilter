#include <benchmark/benchmark.h>
#include "aligned_alloc.hpp"
#include <cstdint>
#include <cstring>
#include <random>
#include <nmmintrin.h>

// ── Helper: fill bitmap with random bits ────────────────────────────────────
static void fill_random(uint64_t* words, size_t n_words, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution dist(0.5);
    for (size_t w = 0; w < n_words; ++w) {
        uint64_t word = 0;
        for (int b = 0; b < 64; ++b)
            if (dist(rng)) word |= (UINT64_C(1) << b);
        words[w] = word;
    }
}

// ── Tier 1: Naive bit-counting loop ─────────────────────────────────────────
//
// Shifts and masks each bit individually. 64 iterations per word.
// Protected from auto-vectorization so GCC cannot turn this into popcnt.
__attribute__((optimize("no-tree-vectorize")))
static uint64_t popcount_naive(const uint64_t* bitmap, size_t n_words) {
    uint64_t total = 0;
    for (size_t i = 0; i < n_words; ++i) {
        uint64_t word = bitmap[i];
        while (word) {
            total += word & 1;
            word >>= 1;
        }
    }
    return total;
}

// ── Tier 2: __builtin_popcountll ────────────────────────────────────────────
//
// Compiler intrinsic — GCC emits popcnt on -mpopcnt targets, but this goes
// through the compiler's own lowering rather than a direct intrinsic call.
__attribute__((optimize("no-tree-vectorize")))
static uint64_t popcount_builtin(const uint64_t* bitmap, size_t n_words) {
    uint64_t total = 0;
    for (size_t i = 0; i < n_words; ++i)
        total += static_cast<uint64_t>(__builtin_popcountll(bitmap[i]));
    return total;
}

// ── Tier 3: _mm_popcnt_u64 (hardware instruction) ──────────────────────────
//
// Direct intrinsic for the POPCNT hardware instruction. Single cycle latency,
// one instruction per word. This is the production path used in query_eval.hpp.
static uint64_t popcount_hw(const uint64_t* bitmap, size_t n_words) {
    uint64_t total = 0;
    for (size_t i = 0; i < n_words; ++i)
        total += _mm_popcnt_u64(bitmap[i]);
    return total;
}

// ── Fixture ─────────────────────────────────────────────────────────────────
class PopcountFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        n_users_ = static_cast<size_t>(state.range(0));
        n_words_ = (n_users_ + 63) / 64;
        bitmap_  = make_aligned_bitmap(n_words_);
        fill_random(bitmap_.get(), n_words_, 0xDEAD'BEEF'CAFE'F00DULL);
    }

    void TearDown(const benchmark::State&) override {
        bitmap_.reset();
    }

protected:
    size_t n_users_ = 0;
    size_t n_words_ = 0;
    AlignedBuffer bitmap_;
};

// ── Benchmarks ──────────────────────────────────────────────────────────────
BENCHMARK_DEFINE_F(PopcountFixture, Naive)(benchmark::State& state) {
    for (auto _ : state) {
        auto count = popcount_naive(bitmap_.get(), n_words_);
        benchmark::DoNotOptimize(count);
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 8);
}

BENCHMARK_DEFINE_F(PopcountFixture, Builtin)(benchmark::State& state) {
    for (auto _ : state) {
        auto count = popcount_builtin(bitmap_.get(), n_words_);
        benchmark::DoNotOptimize(count);
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 8);
}

BENCHMARK_DEFINE_F(PopcountFixture, Hardware)(benchmark::State& state) {
    for (auto _ : state) {
        auto count = popcount_hw(bitmap_.get(), n_words_);
        benchmark::DoNotOptimize(count);
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 8);
}

BENCHMARK_REGISTER_F(PopcountFixture, Naive)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(PopcountFixture, Builtin)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(PopcountFixture, Hardware)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond);

// ── Multi-threaded popcount (ST vs 12-thread at 500M) ───────────────────────
#include "query_eval.hpp"

class MtPopcountFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        n_users_   = static_cast<size_t>(state.range(0));
        n_threads_ = static_cast<unsigned>(state.range(1));
        n_words_   = (n_users_ + 63) / 64;
        bitmap_    = make_aligned_bitmap(n_words_);
        fill_random(bitmap_.get(), n_words_, 0xDEAD'BEEF'CAFE'F00DULL);
    }

    void TearDown(const benchmark::State&) override {
        bitmap_.reset();
    }

protected:
    size_t   n_users_   = 0;
    size_t   n_words_   = 0;
    unsigned n_threads_ = 0;
    AlignedBuffer bitmap_;
};

BENCHMARK_DEFINE_F(MtPopcountFixture, PopcountMT)(benchmark::State& state) {
    for (auto _ : state) {
        auto count = popcount_mt(bitmap_.get(), n_words_, n_threads_);
        benchmark::DoNotOptimize(count);
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 8);
}

BENCHMARK_REGISTER_F(MtPopcountFixture, PopcountMT)
    ->Args({500'000'000, 1})
    ->Args({500'000'000, 2})
    ->Args({500'000'000, 4})
    ->Args({500'000'000, 8})
    ->Args({500'000'000, 12})
    ->Unit(benchmark::kMillisecond);

// main() provided by benchmark::benchmark_main
