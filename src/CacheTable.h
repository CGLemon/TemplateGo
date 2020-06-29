#ifndef CACHETABLE_H_INCLUDE
#define CACHETABLE_H_INCLUDE

#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "NetPipe.h"
#include "config.h"
#include "LZ/LZNetParameters.h"

struct TAResult {
  TAResult() : blackscore(0), whitescore(0) {}
  int blackscore;
  int whitescore;
};

struct NNResult {
  NNResult() : policy_pass(0.0f) {
    policy.fill(0.0f);
    winrate.fill(0.0f);
  }

  std::array<float, NUM_INTERSECTIONS> policy;
  std::array<float, LZ::VALUE_LABELS> winrate;
  float policy_pass;
};

template <typename EvalResult> class CacheTable {
public:
  CacheTable(size_t size = MAX_CACHE_COUNT)
      : m_hits(0), m_lookups(0), m_inserts(0) {
    resize(size);
  }

  bool lookup(std::uint64_t hash, EvalResult &result);
  void insert(std::uint64_t hash, const EvalResult &result);
  void resize(size_t size);

  void dump_stats();

  size_t get_estimated_size();

private:
  static constexpr size_t MAX_CACHE_COUNT = 150000;

  static constexpr size_t MIN_CACHE_COUNT = 6000;

  static constexpr size_t ENTRY_SIZE = sizeof(EvalResult) +
                                       sizeof(std::uint64_t) +
                                       sizeof(std::unique_ptr<EvalResult>);

  std::mutex m_mutex;

  size_t m_size;

  int m_hits;
  int m_lookups;
  int m_inserts;

  struct Entry {
    Entry(const EvalResult &r) : result(r) {}
    EvalResult result;
  };

  std::unordered_map<std::uint64_t, std::unique_ptr<const Entry>> m_cache;

  std::deque<size_t> m_order;
};

template <typename EvalResult>
bool CacheTable<EvalResult>::lookup(std::uint64_t hash, EvalResult &result) {

  std::lock_guard<std::mutex> lock(m_mutex);
  ++m_lookups;

  auto iter = m_cache.find(hash);
  if (iter == m_cache.end()) {
    return false;
  }
  const auto &entry = iter->second;

  ++m_hits;
  result = entry->result;
  return true;
}

template <typename EvalResult>
void CacheTable<EvalResult>::insert(std::uint64_t hash,
                                    const EvalResult &result) {

  std::lock_guard<std::mutex> lock(m_mutex);

  if (m_cache.find(hash) != m_cache.end()) {
    return;
  }

  m_cache.emplace(hash, std::make_unique<Entry>(result));
  m_order.push_back(hash);
  ++m_inserts;

  if (m_order.size() > m_size) {
    m_cache.erase(m_order.front());
    m_order.pop_front();
  }
}

template <typename EvalResult>
void CacheTable<EvalResult>::resize(size_t size) {

  m_size =
      (size > CacheTable::MAX_CACHE_COUNT
           ? CacheTable::MAX_CACHE_COUNT
           : size < CacheTable::MIN_CACHE_COUNT ? CacheTable::MIN_CACHE_COUNT
                                                : size);

  while (m_order.size() > m_size) {
    m_cache.erase(m_order.front());
    m_order.pop_front();
  }
}
#endif
