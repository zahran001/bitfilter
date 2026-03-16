#pragma once
#include <cstdlib>
#include <memory>
#include <new>

inline void* aligned_malloc(size_t size, size_t alignment) {
#ifdef _WIN32
    return _aligned_malloc(size, alignment);
#else
    return std::aligned_alloc(alignment, size);
#endif
}

inline void aligned_free(void* ptr) {
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

struct AlignedDeleter {
    void operator()(void* p) const { aligned_free(p); }
};

// AlignedBuffer: RAII owner for a 64-byte-aligned array of uint64_t.
// Allocated via aligned_malloc, freed via aligned_free — never delete[].
using AlignedBuffer = std::unique_ptr<uint64_t[], AlignedDeleter>;

// Allocate n_words uint64_t values with 64-byte alignment (cache-line perfect).
// Throws std::bad_alloc if the allocation fails.
// Does NOT zero-initialise — caller must memset before use.
inline AlignedBuffer make_aligned_bitmap(size_t n_words) {
    void* raw = aligned_malloc(n_words * sizeof(uint64_t), 64);
    if (!raw) throw std::bad_alloc{};
    return AlignedBuffer(static_cast<uint64_t*>(raw));
}
