#include "query_eval.hpp"
#include <atomic>
#include <thread>
#include <vector>
#include <algorithm>

// ── Multi-threaded eval: over-partitioned chunk dispatch ─────────────────────
//
// Divides the bitmap into 4 × n_threads chunks. Each thread loops on
// atomic fetch_add to claim chunks dynamically. This automatically balances
// P-cores (fast) vs E-cores (slow) — a fast core simply claims more chunks.
//
// Chunk boundaries are cache-line aligned (8 words = 64 bytes) to prevent
// false sharing where two threads write adjacent words in the same line.

void eval_avx2_prefetch_mt(
    const uint64_t* __restrict__ a,
    const uint64_t* __restrict__ b,
    const uint64_t* __restrict__ not_c,
          uint64_t* __restrict__ result,
    size_t n_words,
    unsigned n_threads)
{
    // Fallback: single-threaded path
    if (n_threads <= 1 || n_words == 0) {
        eval_avx2_prefetch(a, b, not_c, result, n_words);
        return;
    }

    constexpr size_t CL_WORDS = 8;  // 64 bytes / 8 bytes per word

    // Over-partition: 4× the thread count for dynamic load balancing
    unsigned n_chunks = 4 * n_threads;

    // If there aren't enough words for that many chunks, reduce
    if (n_words < static_cast<size_t>(n_chunks) * CL_WORDS)
        n_chunks = std::max(1u, static_cast<unsigned>(n_words / CL_WORDS));

    const size_t chunk_words = (n_words / n_chunks / CL_WORDS) * CL_WORDS;

    // Edge case: if chunk_words is 0 after alignment, just run single-threaded
    if (chunk_words == 0) {
        eval_avx2_prefetch(a, b, not_c, result, n_words);
        return;
    }

    std::atomic<unsigned> next_chunk{0};

    auto worker = [&]() {
        while (true) {
            unsigned c = next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (c >= n_chunks) break;

            size_t begin = c * chunk_words;
            size_t end   = (c == n_chunks - 1) ? n_words : begin + chunk_words;
            size_t count = end - begin;

            eval_avx2_prefetch(a + begin, b + begin, not_c + begin,
                               result + begin, count);
        }
    };

    // Launch n_threads-1 workers; main thread participates
    std::vector<std::jthread> threads;
    threads.reserve(n_threads - 1);
    for (unsigned t = 1; t < n_threads; ++t)
        threads.emplace_back(worker);
    worker();  // main thread participates
    // jthread destructor joins automatically
}

// ── Multi-threaded popcount with thread-local reduction ─────────────────────
//
// Each thread accumulates a local popcount in a cache-line-padded slot,
// avoiding false sharing. After all threads join, the main thread sums
// the array — no atomics on the hot path.

struct alignas(64) PaddedCounter {
    uint64_t value = 0;
};

uint64_t popcount_mt(const uint64_t* bitmap, size_t n_words, unsigned n_threads)
{
    // Fallback: single-threaded path
    if (n_threads <= 1 || n_words == 0)
        return popcount(bitmap, n_words);

    constexpr size_t CL_WORDS = 8;

    unsigned n_chunks = 4 * n_threads;
    if (n_words < static_cast<size_t>(n_chunks) * CL_WORDS)
        n_chunks = std::max(1u, static_cast<unsigned>(n_words / CL_WORDS));

    const size_t chunk_words = (n_words / n_chunks / CL_WORDS) * CL_WORDS;

    if (chunk_words == 0)
        return popcount(bitmap, n_words);

    // One padded counter per thread to avoid false sharing
    std::vector<PaddedCounter> counters(n_threads);
    std::atomic<unsigned> next_chunk{0};

    auto worker = [&](unsigned thread_id) {
        uint64_t local = 0;
        while (true) {
            unsigned c = next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (c >= n_chunks) break;

            size_t begin = c * chunk_words;
            size_t end   = (c == n_chunks - 1) ? n_words : begin + chunk_words;

            for (size_t i = begin; i < end; ++i)
                local += _mm_popcnt_u64(bitmap[i]);
        }
        counters[thread_id].value = local;
    };

    std::vector<std::jthread> threads;
    threads.reserve(n_threads - 1);
    for (unsigned t = 1; t < n_threads; ++t)
        threads.emplace_back(worker, t);
    worker(0);  // main thread = slot 0

    // Sum after all threads have joined
    uint64_t total = 0;
    for (unsigned t = 0; t < n_threads; ++t)
        total += counters[t].value;
    return total;
}
