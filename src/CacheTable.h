#ifndef CACHETABLE_H_INCLUDE
#define CACHETABLE_H_INCLUDE

#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <cassert>

#include "config.h"


struct NNResult {
  NNResult() : policy_pass(0.0f), winrate(0.0f) {
    policy.fill(0.0f);
    final_score.fill(0.0f);
    ownership.fill(0.0f);
    winrate_lables.fill(0.0f);
  }

  std::array<float, NUM_INTERSECTIONS> policy;
  std::array<float, NUM_INTERSECTIONS * 2> final_score;
  std::array<float, NUM_INTERSECTIONS> ownership;
  std::array<float, 21> winrate_lables;

  float winrate;
  float policy_pass;
};

template <typename EvalResult>
class CacheTable {
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

  void clear();

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
  std::deque<std::uint64_t> m_order;



  
  enum class State : std::uint8_t {
    IDLE,
    READY,
    LOOKUP,
    INSERT
  }; 

  std::atomic<State> m_state {State::IDLE};
  std::atomic<size_t> lookup_threads{0};


  bool acquire_lookup();
  void quit_lookup();

  bool acquire_insert();
  void idel();
};

template <typename EvalResult>
bool CacheTable<EvalResult>::lookup(std::uint64_t hash,
                                    EvalResult &result) {
  std::lock_guard<std::mutex> lock(m_mutex);
  
  //while (!acquire_lookup()) {}

  bool success = true;
  ++m_lookups;

  const auto iter = m_cache.find(hash);
  if (iter == m_cache.end()) {
    success = false;

  } else {
    const auto &entry = iter->second;

    ++m_hits;

    result = entry->result;
  }

  //quit_lookup();

  return success;
}

template <typename EvalResult>
void CacheTable<EvalResult>::insert(std::uint64_t hash,
                                    const EvalResult &result) {

  std::lock_guard<std::mutex> lock(m_mutex);

  //while (!acquire_insert()) {}

  if (m_cache.find(hash) == m_cache.end()) {
    m_cache.emplace(hash, std::make_unique<Entry>(result));
    m_order.push_back(hash);
    ++m_inserts;

    if (m_order.size() > m_size) {
      m_cache.erase(m_order.front());
      m_order.pop_front();
    }
  }
  //idel();
}

template <typename EvalResult>
void CacheTable<EvalResult>::resize(size_t size) {

  m_size = (size > CacheTable::MAX_CACHE_COUNT ? CacheTable::MAX_CACHE_COUNT : 
            size < CacheTable::MIN_CACHE_COUNT ? CacheTable::MIN_CACHE_COUNT : size);

  std::lock_guard<std::mutex> lock(m_mutex);
  while (m_order.size() > m_size) {
    m_cache.erase(m_order.front());
    m_order.pop_front();
  }
}

template <typename EvalResult> 
void CacheTable<EvalResult>::clear() {

  std::lock_guard<std::mutex> lock(m_mutex);
  if (!m_order.empty()) {
    m_cache.clear();
    m_order.clear();
  }
}

template <typename EvalResult> 
bool CacheTable<EvalResult>::acquire_lookup() {
  auto idle = State::IDLE;
  auto newval = State::LOOKUP;
  auto suceess = m_state.compare_exchange_strong(idle, newval);

  if (suceess) {
    lookup_threads++;
    return suceess;
  }

  idle = State::LOOKUP;
  newval = State::READY;
  suceess = m_state.compare_exchange_strong(idle, newval);

  if (suceess) {
    lookup_threads++;
    auto s = m_state.exchange(State::LOOKUP);
    assert(s == State::READY);
  }
  return suceess;
}

template <typename EvalResult> 
void CacheTable<EvalResult>::quit_lookup() {

  bool success = false;

  while (!success) {
    auto lookup = State::LOOKUP;
    auto newval = State::READY;
    success = m_state.compare_exchange_strong(lookup, newval);
  }

  lookup_threads--;

  if (lookup_threads == 0) {
    auto s = m_state.exchange(State::IDLE);
    assert(s == State::READY);
  } else {
    auto s = m_state.exchange(State::LOOKUP);
    assert(s == State::READY);
  }
}


template <typename EvalResult> 
bool CacheTable<EvalResult>::acquire_insert() {
  auto idle = State::IDLE;
  auto newval = State::INSERT;
  return m_state.compare_exchange_strong(idle, newval);
}

template <typename EvalResult> 
void CacheTable<EvalResult>::idel() {
  m_state.store(State::IDLE);
  assert(lookup_threads.load() == 0);
}
#endif
