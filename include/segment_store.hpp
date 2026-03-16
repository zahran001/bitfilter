#pragma once
#include "aligned_alloc.hpp"
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>

// SegmentStore owns all segment bitmaps.
//
// Every bitmap has exactly n_words_ uint64_t words, covering n_users_ bits.
// Bits [n_users_ .. n_words_*64) are permanently zero — no padding bits
// are ever set, so popcount() never over-counts.
//
// Bit layout: user ID u lives at word u/64, bit u%64 (LSB = user 0).
// This must be consistent with every caller of set_bit() and get_bit().
class SegmentStore {
public:
    // n_users: total population size (e.g. 500'000'000).
    explicit SegmentStore(size_t n_users);

    // Add a zero-initialised bitmap for the given segment name.
    // Throws std::invalid_argument if the name already exists.
    void add_segment(const std::string& name);

    // Raw read-only pointer to segment bitmap. Throws std::out_of_range
    // if the name does not exist. Use this for eval inputs.
    const uint64_t* get_bitmap(const std::string& name) const;

    // Raw mutable pointer. Throws std::out_of_range if name does not exist.
    // Use this in the data generator to write bits.
    uint64_t* get_bitmap_mut(const std::string& name);

    // Number of uint64_t words in every bitmap.
    size_t n_words() const noexcept { return n_words_; }

    // Number of users (bits in the active range of every bitmap).
    size_t n_users() const noexcept { return n_users_; }

private:
    size_t n_users_;
    size_t n_words_;  // = ceil(n_users / 64)
    std::unordered_map<std::string, AlignedBuffer> segments_;
};

// ── Free helpers (header-only, used by DataGen and tests) ─────────────────────

// Set bit for user_id in bitmap. Caller owns the pointer and guarantees
// user_id < n_users for this bitmap.
inline void set_bit(uint64_t* bitmap, size_t user_id) noexcept {
    bitmap[user_id / 64] |= (UINT64_C(1) << (user_id % 64));
}

// Read bit for user_id.
inline bool get_bit(const uint64_t* bitmap, size_t user_id) noexcept {
    return (bitmap[user_id / 64] >> (user_id % 64)) & UINT64_C(1);
}
