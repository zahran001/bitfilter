// Temporary diagnostic: verify chunk distribution across threads.
// Mirrors the over-partitioned dispatch logic from query_eval_mt.cpp.
//
// Two modes:
//   1. No-work: just claim chunks (exposes thread spawn latency)
//   2. With-work: simulate real memory scan per chunk (realistic distribution)
//
// Build:  g++ -std=c++20 -O2 -mavx2 -mpopcnt -o diag_chunks diag_chunks.cpp -lpthread
// Run:    ./diag_chunks
// Delete after verification.

#include <atomic>
#include <thread>
#include <vector>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <algorithm>
#include <numeric>
#include <nmmintrin.h>

// Allocate a real bitmap to simulate memory-bound work
static uint64_t* alloc_bitmap(size_t n_words) {
    void* p = std::aligned_alloc(64, n_words * sizeof(uint64_t));
    auto* bm = static_cast<uint64_t*>(p);
    for (size_t i = 0; i < n_words; ++i)
        bm[i] = 0xDEAD'BEEF'CAFE'F00DULL ^ i;
    return bm;
}

static void run_test(unsigned n_threads, size_t n_words, const uint64_t* bitmap, bool do_work) {
    constexpr size_t CL_WORDS = 8;
    unsigned n_chunks = 4 * n_threads;
    const size_t chunk_words = (n_words / n_chunks / CL_WORDS) * CL_WORDS;

    std::vector<unsigned> chunk_owner(n_chunks, 0xFFFF);
    std::atomic<unsigned> next_chunk{0};

    auto worker = [&](unsigned tid) {
        volatile uint64_t sink = 0;  // prevent dead-code elimination
        while (true) {
            unsigned c = next_chunk.fetch_add(1, std::memory_order_relaxed);
            if (c >= n_chunks) break;
            chunk_owner[c] = tid;

            if (do_work) {
                size_t begin = c * chunk_words;
                size_t end   = (c == n_chunks - 1) ? n_words : begin + chunk_words;
                uint64_t local = 0;
                for (size_t i = begin; i < end; ++i)
                    local += _mm_popcnt_u64(bitmap[i]);
                sink = local;
            }
        }
    };

    std::vector<std::jthread> threads;
    threads.reserve(n_threads - 1);
    for (unsigned t = 1; t < n_threads; ++t)
        threads.emplace_back(worker, t);
    worker(0);
    threads.clear();

    std::vector<unsigned> counts(n_threads, 0);
    for (unsigned c = 0; c < n_chunks; ++c) {
        if (chunk_owner[c] < n_threads)
            counts[chunk_owner[c]]++;
    }

    std::printf("--- %u threads, %u chunks (%.1f MB/chunk) [%s] ---\n",
                n_threads, n_chunks,
                chunk_words * 8.0 / (1024 * 1024),
                do_work ? "WITH WORK" : "NO WORK");

    for (unsigned t = 0; t < n_threads; ++t)
        std::printf("  thread %2u: %3u chunks\n", t, counts[t]);

    unsigned min_c = *std::min_element(counts.begin(), counts.end());
    unsigned max_c = *std::max_element(counts.begin(), counts.end());
    unsigned total = std::accumulate(counts.begin(), counts.end(), 0u);
    std::printf("  min=%u  max=%u  total=%u  imbalance=%.1f%%\n\n",
                min_c, max_c, total,
                100.0 * (max_c - min_c) / static_cast<double>(max_c));
}

int main() {
    constexpr size_t N_USERS = 500'000'000;
    constexpr size_t N_WORDS = (N_USERS + 63) / 64;

    uint64_t* bitmap = alloc_bitmap(N_WORDS);

    std::printf("=== NO-WORK (exposes thread spawn race) ===\n\n");
    for (unsigned t : {2u, 4u, 8u, 12u})
        run_test(t, N_WORDS, bitmap, false);

    std::printf("=== WITH-WORK (realistic popcount scan) ===\n\n");
    for (unsigned t : {2u, 4u, 8u, 12u})
        run_test(t, N_WORDS, bitmap, true);

    std::free(bitmap);
    return 0;
}
