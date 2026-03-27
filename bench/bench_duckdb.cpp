// bench_duckdb.cpp — BitFilter dense bitmap vs DuckDB columnar SQL engine
//
// Runs the equivalent A AND B AND NOT C query through DuckDB's SQL engine
// to measure the abstraction overhead of a general-purpose columnar database
// versus a purpose-built bitmap engine.
//
// DuckDB uses its own SIMD optimizations internally — the point is to show
// the cost of parsing SQL, building a query plan, managing columnar storage,
// and producing results through a general-purpose pipeline.
//
// Links against prebuilt libduckdb.so (v1.5.1) — the amalgamation source
// OOMs during compilation on machines with <12 GB RAM.

#include "duckdb.hpp"

#include <benchmark/benchmark.h>
#include "../include/aligned_alloc.hpp"
#include "../include/query_eval.hpp"

#include <cstring>
#include <random>
#include <string>

// ── Shared constants ─────────────────────────────────────────────────────────

static constexpr size_t N_USERS = 500'000'000ULL;

static double bp_to_density(int64_t bp) { return static_cast<double>(bp) / 10000.0; }

// ── BitFilter (dense bitmap) benchmark ───────────────────────────────────────
// Duplicated from bench_croaring.cpp so this target is self-contained.

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
    ->Arg(1000)    // 10%
    ->Arg(5000)    // 50%
    ->Unit(benchmark::kMillisecond);

// ── DuckDB benchmark ────────────────────────────────────────────────────────
//
// Strategy: create an in-memory DuckDB database, populate a table with boolean
// columns using DuckDB's own random() function (seeded via setseed()), then
// time the SELECT COUNT(*) WHERE a AND b AND NOT c query.
//
// Setup is expensive (~minutes for 500M rows) but runs once per fixture.
// Only the query execution is timed.

class DuckDBFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        double density = bp_to_density(state.range(0));

        db_  = std::make_unique<duckdb::DuckDB>(nullptr); // in-memory
        con_ = std::make_unique<duckdb::Connection>(*db_);

        // Create table and populate with random boolean data.
        // DuckDB's random() returns [0,1) — compare against density threshold.
        // Use setseed() for reproducibility.
        con_->Query("SELECT setseed(0.42)");

        // Create table with boolean columns based on density threshold.
        // INSERT INTO ... SELECT is the fastest bulk-load path in DuckDB.
        // range() generates the row IDs; random() < density gives booleans.
        std::string sql =
            "CREATE TABLE segments AS "
            "SELECT "
            "  random() < " + std::to_string(density) + " AS a, "
            "  random() < " + std::to_string(density) + " AS b, "
            "  random() < " + std::to_string(density) + " AS c "
            "FROM range(0, " + std::to_string(N_USERS) + ")";

        auto result = con_->Query(sql);
        if (result->HasError()) {
            fprintf(stderr, "DuckDB setup failed: %s\n", result->GetError().c_str());
            std::abort();
        }
    }

    void TearDown(const benchmark::State&) override {
        con_.reset();
        db_.reset();
    }

protected:
    std::unique_ptr<duckdb::DuckDB>     db_;
    std::unique_ptr<duckdb::Connection> con_;
};

BENCHMARK_DEFINE_F(DuckDBFixture, CountWhereAndAndNot)(benchmark::State& state) {
    for (auto _ : state) {
        auto result = con_->Query(
            "SELECT COUNT(*) FROM segments WHERE a AND b AND NOT c");
        benchmark::DoNotOptimize(result->GetValue(0, 0));
    }
}

BENCHMARK_REGISTER_F(DuckDBFixture, CountWhereAndAndNot)
    ->Arg(1000)    // 10%
    ->Arg(5000)    // 50%
    ->Unit(benchmark::kMillisecond)
    ->Iterations(5);  // DuckDB is slow enough that 5 iterations suffices

// main() provided by benchmark::benchmark_main
