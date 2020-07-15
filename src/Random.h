#ifndef RANDON_H_INCLUDE
#define RANDON_H_INCLUDE

#include <cstdint>
#include <limits>
#include <random>
#include <thread>
#include <time.h>

/*
 https://github.com/lemire/testingRNG
 */


static constexpr std::uint64_t THREADS_SEED = 0;
static constexpr std::uint64_t TIME_SEED = 1;

namespace random_utils {

static inline std::uint64_t splitmix64(std::uint64_t z) {
  /*
   The parameter detail are from
   https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h
   */

  z += 0x9e3779b97f4a7c15;
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

static inline std::uint64_t rotl(const std::uint64_t x, const int k) {
  return (x << k) | (x >> (64 - k));
}

} // namespace random_utils


enum class random_t {
  SplitMix_64,
  XorShiro128Plus
};

template<random_t RandomType>
class Random {
public:
  static constexpr size_t SEED_SZIE = 2;

  Random() = delete;

  Random(std::uint64_t seed) { seed_init(seed); }
  
  static Random &get_Rng(const std::uint64_t seed = THREADS_SEED);

  std::uint64_t randuint64();

  template<int Range>
  std::uint32_t randfix() {
    static_assert(0 < Range && Range < std::numeric_limits<std::uint32_t>::max(),
                "randfix out of range");
    return randuint64() % Range;
  }
 
  using result_type = std::uint64_t;

  constexpr static result_type min() {
    return std::numeric_limits<result_type>::min();
  }
  constexpr static result_type max() {
    return std::numeric_limits<result_type>::max();
  }

  result_type operator()() { return randuint64(); }

private:
  void seed_init(std::uint64_t);

  static thread_local std::array<std::uint64_t, SEED_SZIE> m_seeds;
};


template<random_t T>
void Random<T>::seed_init(std::uint64_t seed) {
  if (seed == THREADS_SEED) {
    auto thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
    seed = static_cast<std::uint64_t>(thread_id);
  } else if (seed == TIME_SEED) {
    auto get_time = std::time(NULL);
    seed = static_cast<std::uint64_t>(get_time);
  }
  const size_t size = m_seeds.size(); 
  for (auto i = size_t{0}; i < size; ++i) {
    seed = random_utils::splitmix64(seed);
    m_seeds[i] = seed;
  }
}

template<random_t T>
Random<T> &Random<T>::get_Rng(const std::uint64_t seed) {
  static thread_local Random s_rng{seed};
  return s_rng;
}
#endif
