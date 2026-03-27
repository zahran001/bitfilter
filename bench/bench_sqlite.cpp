// bench_sqlite.cpp — BitFilter dense bitmap vs SQLite row-store
//
// SQLite is a row-store: every row is fetched, parsed, and evaluated
// individually. This benchmark shows the fundamental cost of row-by-row
// evaluation vs bitmap-level parallelism.
//
// Runs a scaling sweep at 5M, 10M, 50M rows to prove O(N) linearity,
// then extrapolates to 500M. BitFilter runs at the same scale points
// for direct comparison.
//
// Density is fixed at 50% — worst case for both engines (maximum work).

#include <benchmark/benchmark.h>
#include "aligned_alloc.hpp"
#include "query_eval.hpp"

#include <sqlite3.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>

// ── Shared constants ─────────────────────────────────────────────────────────

// Scale points: 5M, 10M, 50M rows
// Encoded as millions so Args() values are readable.
static constexpr double DENSITY = 0.5;

// ── BitFilter (dense bitmap) benchmark at variable scale ─────────────────────

class BitFilterScaleFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        size_t n_users = static_cast<size_t>(state.range(0)) * 1'000'000ULL;
        n_words_ = (n_users + 63) / 64;

        a_      = make_aligned_bitmap(n_words_);
        b_      = make_aligned_bitmap(n_words_);
        not_c_  = make_aligned_bitmap(n_words_);
        result_ = make_aligned_bitmap(n_words_);

        std::memset(a_.get(),      0, n_words_ * sizeof(uint64_t));
        std::memset(b_.get(),      0, n_words_ * sizeof(uint64_t));
        std::memset(not_c_.get(),  0, n_words_ * sizeof(uint64_t));
        std::memset(result_.get(), 0, n_words_ * sizeof(uint64_t));

        fill_bitmap(a_.get(),     n_users, DENSITY, 0xAAAA);
        fill_bitmap(b_.get(),     n_users, DENSITY, 0xBBBB);
        fill_bitmap(not_c_.get(), n_users, DENSITY, 0xCCCC);
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

BENCHMARK_DEFINE_F(BitFilterScaleFixture, EvalAvx2Prefetch)(benchmark::State& state) {
    for (auto _ : state) {
        eval_avx2_prefetch(a_.get(), b_.get(), not_c_.get(), result_.get(), n_words_);
        benchmark::ClobberMemory();
    }
    state.SetBytesProcessed(
        static_cast<int64_t>(state.iterations()) *
        static_cast<int64_t>(n_words_) * 4 * 8);
    state.SetLabel(std::to_string(state.range(0)) + "M users");
}

BENCHMARK_REGISTER_F(BitFilterScaleFixture, EvalAvx2Prefetch)
    ->Arg(5)       // 5M
    ->Arg(10)      // 10M
    ->Arg(50)      // 50M
    ->Unit(benchmark::kMillisecond);

// ── SQLite benchmark ────────────────────────────────────────────────────────
//
// Strategy: create an in-memory SQLite database, bulk-insert rows inside a
// transaction (critical — without BEGIN/COMMIT, SQLite does one fsync per
// INSERT even for :memory:), then time the SELECT COUNT(*) query.
//
// The query is:
//   SELECT COUNT(*) FROM segments WHERE a = 1 AND b = 1 AND c = 0
//
// No indexes — this is a full-table scan, which is the fair comparison
// against a bitmap engine that always scans everything.

static void exec_or_die(sqlite3* db, const char* sql) {
    char* err = nullptr;
    if (sqlite3_exec(db, sql, nullptr, nullptr, &err) != SQLITE_OK) {
        fprintf(stderr, "SQLite error: %s\nSQL: %s\n", err, sql);
        sqlite3_free(err);
        std::abort();
    }
}

class SQLiteFixture : public benchmark::Fixture {
public:
    void SetUp(const benchmark::State& state) override {
        size_t n_rows = static_cast<size_t>(state.range(0)) * 1'000'000ULL;

        if (sqlite3_open(":memory:", &db_) != SQLITE_OK) {
            fprintf(stderr, "SQLite open failed: %s\n", sqlite3_errmsg(db_));
            std::abort();
        }

        // WAL mode + large page size for maximum in-memory performance
        exec_or_die(db_, "PRAGMA journal_mode = OFF");
        exec_or_die(db_, "PRAGMA synchronous = OFF");
        exec_or_die(db_, "PRAGMA page_size = 65536");

        exec_or_die(db_, "CREATE TABLE segments (a INTEGER, b INTEGER, c INTEGER)");

        // Bulk insert inside a single transaction with prepared statement
        exec_or_die(db_, "BEGIN");

        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "INSERT INTO segments VALUES (?, ?, ?)", -1, &stmt, nullptr);

        std::mt19937_64 rng_a(0xAAAA);
        std::mt19937_64 rng_b(0xBBBB);
        std::mt19937_64 rng_c(0xCCCC);
        std::bernoulli_distribution dist(DENSITY);

        for (size_t i = 0; i < n_rows; ++i) {
            sqlite3_bind_int(stmt, 1, dist(rng_a) ? 1 : 0);
            sqlite3_bind_int(stmt, 2, dist(rng_b) ? 1 : 0);
            sqlite3_bind_int(stmt, 3, dist(rng_c) ? 1 : 0);
            sqlite3_step(stmt);
            sqlite3_reset(stmt);
        }
        sqlite3_finalize(stmt);

        exec_or_die(db_, "COMMIT");
    }

    void TearDown(const benchmark::State&) override {
        if (db_) {
            sqlite3_close(db_);
            db_ = nullptr;
        }
    }

protected:
    sqlite3* db_ = nullptr;
};

BENCHMARK_DEFINE_F(SQLiteFixture, CountWhereAndAndNot)(benchmark::State& state) {
    for (auto _ : state) {
        sqlite3_stmt* stmt = nullptr;
        sqlite3_prepare_v2(db_,
            "SELECT COUNT(*) FROM segments WHERE a = 1 AND b = 1 AND c = 0",
            -1, &stmt, nullptr);
        sqlite3_step(stmt);
        int64_t count = sqlite3_column_int64(stmt, 0);
        benchmark::DoNotOptimize(count);
        sqlite3_finalize(stmt);
    }
    state.SetLabel(std::to_string(state.range(0)) + "M rows");
}

BENCHMARK_REGISTER_F(SQLiteFixture, CountWhereAndAndNot)
    ->Arg(5)       // 5M
    ->Arg(10)      // 10M
    ->Arg(50)      // 50M
    ->Unit(benchmark::kMillisecond)
    ->Iterations(3);  // SQLite is slow — 3 iterations suffices

// main() provided by benchmark::benchmark_main
