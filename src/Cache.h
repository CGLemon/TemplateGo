#ifndef CACHE_H_INCLUDE
#define CACHE_H_INCLUDE

#include <array>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <atomic>
#include <cassert>

#include "config.h"
#include "Utils.h"

struct NNResult {
    NNResult() {
        policy.fill(0.0f);
        ownership.fill(0.0f);
        // multi_labeled.fill(0.0f);
    }

    // float winrate{0.0f};
    float policy_pass{0.0f};
    float final_score{0.0f};

    float alpha{0.0f};
    float beta{0.0f};
    float gamma{0.0f};

    std::array<float, NUM_INTERSECTIONS> policy;
    std::array<float, NUM_INTERSECTIONS> ownership;
    // std::array<float, 21> multi_labeled;
};

template <typename EvalResult>
class Cache {
public:
    Cache(size_t size = MAX_CACHE_COUNT)
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
};

template <typename EvalResult>
bool Cache<EvalResult>::lookup(std::uint64_t hash,
                                    EvalResult &result) {
    std::lock_guard<std::mutex> lock(m_mutex);
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
    return success;
}

template <typename EvalResult>
void Cache<EvalResult>::insert(std::uint64_t hash,
                                    const EvalResult &result) {

    std::lock_guard<std::mutex> lock(m_mutex);

    if (m_cache.find(hash) == m_cache.end()) {
        m_cache.emplace(hash, std::make_unique<Entry>(result));
        m_order.emplace_back(hash);
        ++m_inserts;

        if (m_order.size() > m_size) {
            m_cache.erase(m_order.front());
            m_order.pop_front();
        }
    }
}

template <typename EvalResult>
void Cache<EvalResult>::resize(size_t size) {

    m_size = (size > Cache::MAX_CACHE_COUNT ? Cache::MAX_CACHE_COUNT : 
              size < Cache::MIN_CACHE_COUNT ? Cache::MIN_CACHE_COUNT : size);

    std::lock_guard<std::mutex> lock(m_mutex);
    while (m_order.size() > m_size) {
        m_cache.erase(m_order.front());
        m_order.pop_front();
    }
}

template <typename EvalResult> 
void Cache<EvalResult>::clear() {

    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_order.empty()) {
        m_cache.clear();
        m_order.clear();
    }
}

template <typename EvalResult>
size_t Cache<EvalResult>::get_estimated_size() {
    return m_order.size() * Cache::ENTRY_SIZE;
}

template <typename EvalResult> 
void Cache<EvalResult>::dump_stats() {
    Utils::auto_printf("NNCache: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, %lu size\n",
                       m_hits, m_lookups, 100. * m_hits / (m_lookups + 1), m_inserts,
                       m_cache.size());
}
#endif
