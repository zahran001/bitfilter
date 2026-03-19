#include <benchmark/benchmark.h>
#include "aligned_alloc.hpp"
#include "query_eval.hpp"
#include <cstring>
#include <random>

// ── Helper: fill bitmap with random bits ────────────────────────────────────
// Same bernoulli approach as the test suite. Not on the hot path.
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

// ── Fixture: allocate and populate bitmaps once per benchmark case ───────────
//
// The fixture owns four aligned buffers (a, b, not_c, result).
// Setup runs once per (function × n_users) combination, not per iteration.
// Fixed seeds ensure every run exercises identical bit patterns.
class EvalFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        n_users_ = static_cast<size_t>(state.range(0));
        n_words_ = (n_users_ + 63) / 64;

        a_      = make_aligned_bitmap(n_words_);
        b_      = make_aligned_bitmap(n_words_);
        not_c_  = make_aligned_bitmap(n_words_);
        result_ = make_aligned_bitmap(n_words_);

        fill_random(a_.get(),     n_words_, 0xAAAA'AAAA'AAAA'AAAAULL);
        fill_random(b_.get(),     n_words_, 0xBBBB'BBBB'BBBB'BBBBULL);
        fill_random(not_c_.get(), n_words_, 0xCCCC'CCCC'CCCC'CCCCULL);
        std::memset(result_.get(), 0, n_words_ * sizeof(uint64_t));
    }

    void TearDown(const benchmark::State&) override {
        a_.reset();
        b_.reset();
        not_c_.reset();
        result_.reset();
    }

protected:
    size_t n_users_ = 0;
    size_t n_words_ = 0;
    AlignedBuffer a_, b_, not_c_, result_;
};

// ── Scalar baseline ─────────────────────────────────────────────────────────
BENCHMARK_DEFINE_F(EvalFixture, Scalar)(benchmark::State& state) {
    for (auto _ : state) {
        eval_scalar(a_.get(), b_.get(), not_c_.get(), result_.get(), n_words_);
        benchmark::ClobberMemory();
    }
    // 3 bitmaps read + 1 written = 4 × n_words × 8 bytes per iteration
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 4 * 8);
}

// ── AVX2 primary path ───────────────────────────────────────────────────────
BENCHMARK_DEFINE_F(EvalFixture, Avx2)(benchmark::State& state) {
    for (auto _ : state) {
        eval_avx2(a_.get(), b_.get(), not_c_.get(), result_.get(), n_words_);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 4 * 8);
}

// ── Register with three population sizes ────────────────────────────────────
//
//   1M users   (~500 KB total bitmap data) — fits comfortably in L3
//   10M users  (~5 MB)                     — exceeds L3, starts hitting DRAM
//   500M users (~250 MB)                   — full production scale, pure DRAM
//
// Unit is milliseconds so large-scale numbers are readable.

BENCHMARK_REGISTER_F(EvalFixture, Scalar)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond);

BENCHMARK_REGISTER_F(EvalFixture, Avx2)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond);

// ── AVX2 2x unrolled ────────────────────────────────────────────────────────
BENCHMARK_DEFINE_F(EvalFixture, Avx2Unroll2)(benchmark::State& state) {
    for (auto _ : state) {
        eval_avx2_unroll2(a_.get(), b_.get(), not_c_.get(), result_.get(), n_words_);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 4 * 8);
}

BENCHMARK_REGISTER_F(EvalFixture, Avx2Unroll2)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond);

// ── AVX2 4x unrolled ────────────────────────────────────────────────────────
BENCHMARK_DEFINE_F(EvalFixture, Avx2Unroll4)(benchmark::State& state) {
    for (auto _ : state) {
        eval_avx2_unroll4(a_.get(), b_.get(), not_c_.get(), result_.get(), n_words_);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 4 * 8);
}

BENCHMARK_REGISTER_F(EvalFixture, Avx2Unroll4)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond);

// ── AVX2 with prefetching ────────────────────────────────────────────────────
BENCHMARK_DEFINE_F(EvalFixture, Avx2Prefetch)(benchmark::State& state) {
    for (auto _ : state) {
        eval_avx2_prefetch(a_.get(), b_.get(), not_c_.get(), result_.get(), n_words_);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 4 * 8);
}

BENCHMARK_REGISTER_F(EvalFixture, Avx2Prefetch)
    ->Arg(1'000'000)
    ->Arg(10'000'000)
    ->Arg(500'000'000)
    ->Unit(benchmark::kMillisecond);

// main() provided by benchmark::benchmark_main
