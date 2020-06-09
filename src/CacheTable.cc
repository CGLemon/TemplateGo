#include "CacheTable.h"

#include <functional>
#include <memory>

template <typename EvalResult>
constexpr size_t CacheTable<EvalResult>::MAX_CACHE_COUNT;

template <typename EvalResult>
constexpr size_t CacheTable<EvalResult>::MIN_CACHE_COUNT;

template <typename EvalResult>
constexpr size_t CacheTable<EvalResult>::ENTRY_SIZE;

template <typename EvalResult>
size_t CacheTable<EvalResult>::get_estimated_size() {
  return m_order.size() * CacheTable::ENTRY_SIZE;
}

template <typename EvalResult> void CacheTable<EvalResult>::dump_stats() {
  printf("NNCache: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, %lu size\n",
         m_hits, m_lookups, 100. * m_hits / (m_lookups + 1), m_inserts,
         m_cache.size());
}
