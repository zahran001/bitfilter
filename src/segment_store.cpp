#include "segment_store.hpp"
#include <cstring>
#include <stdexcept>

SegmentStore::SegmentStore(size_t n_users)
    : n_users_(n_users)
    , n_words_((n_users + 63) / 64)  // ceiling division: covers all bits
{}

void SegmentStore::add_segment(const std::string& name) {
    if (segments_.count(name))
        throw std::invalid_argument("segment already exists: " + name);

    AlignedBuffer buf = make_aligned_bitmap(n_words_);
    // Zero-init is mandatory: aligned_alloc does not initialise memory.
    // Unzeroed padding bits would corrupt popcount and query results.
    std::memset(buf.get(), 0, n_words_ * sizeof(uint64_t));
    segments_.emplace(name, std::move(buf));
}

const uint64_t* SegmentStore::get_bitmap(const std::string& name) const {
    return segments_.at(name).get();  // throws std::out_of_range on miss
}

uint64_t* SegmentStore::get_bitmap_mut(const std::string& name) {
    return segments_.at(name).get();
}
