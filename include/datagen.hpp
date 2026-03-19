#pragma once
#include "segment_store.hpp"
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>

// DataGen — reproducible synthetic bitmap population.
//
// Design choices:
//
//   Engine: std::mt19937_64 — a 64-bit Mersenne Twister. Produces high-quality
//   pseudorandom bits with a period of 2^19937-1. We use the 64-bit variant
//   because we work with uint64_t words; the 32-bit variant would require two
//   calls per word.
//
//   Distribution: std::bernoulli_distribution(density) — fires true with
//   probability `density` for each user. This correctly models the statistical
//   independence of segment membership across users.
//
//   Seed: caller-supplied, defaulting to 42. Use the same seed in tests and
//   benchmarks so every run exercises identical bit patterns. Non-reproducible
//   seeds make performance regressions indistinguishable from input variance.
//
//   Why not fill whole words at once?
//   Generating one bit per user via bernoulli is ~3x slower than filling
//   uint64_t words directly, but it is exactly correct for any density and any
//   n_users — including when n_users is not a multiple of 64. A word-fill
//   approach requires careful masking of the last word to avoid setting padding
//   bits. For Week 1 correctness work, clarity beats speed in the generator.
//   Week 2+ can introduce a faster word-fill variant if generator time shows up
//   in profiles.

class DataGen {
public:
    // Populate `segment_name` in `store` with bits set at the given density.
    //
    // density: fraction of users with the bit set. 0.0 = all zeros, 1.0 = all
    // ones. Values outside [0,1] throw std::invalid_argument.
    //
    // seed: RNG seed. Use the same value across all callers that need
    // comparable inputs. Default 42.
    //
    // The segment must already exist in the store (call add_segment first).
    // This function only writes — it does not create the segment.
    static void populate(
        SegmentStore&      store,
        const std::string& segment_name,
        double             density,
        uint64_t           seed = 42)
    {
        if (density < 0.0 || density > 1.0)
            throw std::invalid_argument("density must be in [0.0, 1.0]");

        uint64_t* bm   = store.get_bitmap_mut(segment_name);
        size_t n_users = store.n_users();

        std::mt19937_64                    rng(seed);
        std::bernoulli_distribution        dist(density);

        for (size_t uid = 0; uid < n_users; ++uid) {
            if (dist(rng))
                set_bit(bm, uid);
        }
    }
};
