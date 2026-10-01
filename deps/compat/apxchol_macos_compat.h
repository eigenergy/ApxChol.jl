// Force-included on macOS while the pinned apxchol revision predates its
// macOS portability fix: the Linux-only madvise advice constants become
// MADV_NORMAL and the Linux-only lazy-pool THP probe reports false. Deleted
// once the package builds against an apxchol revision that guards them.
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
