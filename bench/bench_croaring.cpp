// bench_croaring.cpp — BitFilter dense bitmap vs CRoaring for A AND B AND NOT C
//
// Runs the same boolean query at multiple segment densities to show the
// crossover point: dense bitmaps dominate at high density; CRoaring wins
// when segments are sparse and compression kicks in.

#include <benchmark/benchmark.h>
#include "aligned_alloc.hpp"
#include "query_eval.hpp"
#include "segment_store.hpp"
#include "datagen.hpp"

#include <roaring/roaring.h>

#include <cstring>
#include <random>
#include <vector>

// ── Shared constants ─────────────────────────────────────────────────────────

static constexpr size_t N_USERS = 500'000'000ULL;

// Densities to test: 0.01%, 0.1%, 1%, 10%, 50%
// Encoded as basis points (1 bp = 0.01%) so we can pass integers via Args().
// 1 = 0.01%, 10 = 0.1%, 100 = 1%, 1000 = 10%, 5000 = 50%
static double bp_to_density(int64_t bp) { return static_cast<double>(bp) / 10000.0; }

// ── BitFilter (dense bitmap) benchmark ───────────────────────────────────────

class BitFilterFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        double density = bp_to_density(state.range(0));
        n_words_ = (N_USERS + 63) / 64;

        a_      = make_aligned_bitmap(n_words_);
        b_      = make_aligned_bitmap(n_words_);
        not_c_  = make_aligned_bitmap(n_words_);
        result_ = make_aligned_bitmap(n_words_);

        std::memset(a_.get(),      0, n_words_ * sizeof(uint64_t));
        std::memset(b_.get(),      0, n_words_ * sizeof(uint64_t));
        std::memset(not_c_.get(),  0, n_words_ * sizeof(uint64_t));
        std::memset(result_.get(), 0, n_words_ * sizeof(uint64_t));

        fill_bitmap(a_.get(),     N_USERS, density, 0xAAAA);
        fill_bitmap(b_.get(),     N_USERS, density, 0xBBBB);
        fill_bitmap(not_c_.get(), N_USERS, density, 0xCCCC);
    }

    void TearDown(const benchmark::State&) override {
        a_.reset(); b_.reset(); not_c_.reset(); result_.reset();
    }

protected:
    size_t n_words_ = 0;
    AlignedBuffer a_, b_, not_c_, result_;

    // Fill a raw bitmap at the given density using bernoulli distribution.
    static void fill_bitmap(uint64_t* bm, size_t n_users, double density, uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::bernoulli_distribution dist(density);
        for (size_t uid = 0; uid < n_users; ++uid) {
            if (dist(rng))
                bm[uid / 64] |= (UINT64_C(1) << (uid % 64));
        }
    }
};

BENCHMARK_DEFINE_F(BitFilterFixture, EvalAvx2Prefetch)(benchmark::State& state) {
    for (auto _ : state) {
        eval_avx2_prefetch(a_.get(), b_.get(), not_c_.get(), result_.get(), n_words_);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 4 * 8);
}

BENCHMARK_REGISTER_F(BitFilterFixture, EvalAvx2Prefetch)
    ->Arg(1)       // 0.01%
    ->Arg(10)      // 0.1%
    ->Arg(100)     // 1%
    ->Arg(1000)    // 10%
    ->Arg(5000)    // 50%
    ->Unit(benchmark::kMillisecond);

// ── CRoaring benchmark ──────────────────────────────────────────────────────

class CRoaringFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        double density = bp_to_density(state.range(0));

        ra_ = roaring_bitmap_create();
        rb_ = roaring_bitmap_create();
        rc_ = roaring_bitmap_create();

        fill_roaring(ra_, N_USERS, density, 0xAAAA);
        fill_roaring(rb_, N_USERS, density, 0xBBBB);
        fill_roaring(rc_, N_USERS, density, 0xCCCC);

        // Optimize internal representation (run-compress where beneficial)
        roaring_bitmap_run_optimize(ra_);
        roaring_bitmap_run_optimize(rb_);
        roaring_bitmap_run_optimize(rc_);
    }

    void TearDown(const benchmark::State&) override {
        roaring_bitmap_free(ra_);
        roaring_bitmap_free(rb_);
        roaring_bitmap_free(rc_);
    }

protected:
    roaring_bitmap_t* ra_ = nullptr;
    roaring_bitmap_t* rb_ = nullptr;
    roaring_bitmap_t* rc_ = nullptr;

    // Fill a Roaring bitmap with the same RNG sequence as BitFilter's dense path
    // so both engines operate on identical logical sets.
    static void fill_roaring(roaring_bitmap_t* bm, size_t n_users, double density, uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::bernoulli_distribution dist(density);

        // Batch user IDs and add in bulk for efficiency
        std::vector<uint32_t> ids;
        ids.reserve(static_cast<size_t>(static_cast<double>(n_users) * density * 1.1));

        for (size_t uid = 0; uid < n_users; ++uid) {
            if (dist(rng))
                ids.push_back(static_cast<uint32_t>(uid));
        }
        roaring_bitmap_add_many(bm, ids.size(), ids.data());
    }
};

BENCHMARK_DEFINE_F(CRoaringFixture, AndAndNot)(benchmark::State& state) {
    for (auto _ : state) {
        // A AND B
        roaring_bitmap_t* ab = roaring_bitmap_and(ra_, rb_);
        // (A AND B) AND NOT C
        roaring_bitmap_t* result = roaring_bitmap_andnot(ab, rc_);

        benchmark::DoNotOptimize(roaring_bitmap_get_cardinality(result));

        roaring_bitmap_free(result);
        roaring_bitmap_free(ab);
    }
    // CRoaring doesn't have a fixed memory footprint per iteration,
    // so we skip SetBytesProcessed — compare on latency only.
}

BENCHMARK_REGISTER_F(CRoaringFixture, AndAndNot)
    ->Arg(1)       // 0.01%
    ->Arg(10)      // 0.1%
    ->Arg(100)     // 1%
    ->Arg(1000)    // 10%
    ->Arg(5000)    // 50%
    ->Unit(benchmark::kMillisecond);

// main() provided by benchmark::benchmark_main
