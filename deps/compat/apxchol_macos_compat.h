// Force-included on macOS while the pinned apxchol revision predates its
// macOS portability fix: the Linux-only madvise advice constants become
// MADV_NORMAL, the Linux-only lazy-pool THP probe reports false, and Xcode 16's
// libc++ (no std::ranges::iota) gets the one overload apxchol calls. Deleted
// once the package builds against an apxchol revision that includes the fix.
#pragma once
#include <sys/mman.h>
#ifndef MADV_HUGEPAGE
#define MADV_HUGEPAGE MADV_NORMAL
#endif
#ifndef MADV_NOHUGEPAGE
#define MADV_NOHUGEPAGE MADV_NORMAL
#endif
#ifndef MADV_POPULATE_WRITE
#define MADV_POPULATE_WRITE MADV_NORMAL
#endif
#ifndef __linux__
namespace apxchol::detail {
inline bool lazy_pool_uses_thp() { return false; }
}
#endif

// Xcode 16 libc++ lacks std::ranges::iota; apxchol calls iota(range, value).
#include <version>
#if !defined(__cpp_lib_ranges_iota)
#include <iterator>
#include <numeric>
#include <ranges>
namespace std::ranges {
struct apxchol_compat_iota_fn {
    template <class R, class T>
    constexpr void operator()(R&& r, T value) const {
        std::iota(std::ranges::begin(r), std::ranges::end(r), value);
    }
};
inline constexpr apxchol_compat_iota_fn iota{};
}  // namespace std::ranges
#endif
