#ifndef RANDON_H_INCLUDE
#define RANDON_H_INCLUDE

#include <cstdint>
#include <limits>
#include <random>
#include <thread>

/*
 * https://github.com/lemire/testingRNG
 */

static constexpr std::uint64_t CFG_DEFAULT = 0;
static constexpr std::uint64_t THREADS_SEED = 1;
static constexpr std::uint64_t TIME_SEED = 2;

enum class random_t {
  SplitMix_64,
  XoroShiro128Plus,
};

template<random_t RandomType>
class Random {
public:
  Random() = delete;

  Random(std::uint64_t seed);
  
  static Random &get_Rng(const std::uint64_t seed = THREADS_SEED);

  std::uint64_t randuint64();

  template<int Range>
  std::uint32_t randfix() {
    static_assert(0 < Range && Range < std::numeric_limits<std::uint32_t>::max(),
                "randfix out of range?\n");
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
  static constexpr size_t SEED_SZIE = 2;

  static thread_local std::uint64_t m_seeds[SEED_SZIE];

  void seed_init(std::uint64_t);
};

template<random_t T>
Random<T>::Random(std::uint64_t seed) {
  seed_init(seed);
}

template<random_t T>
Random<T> &Random<T>::get_Rng(const std::uint64_t seed) {
  static thread_local Random s_rng{seed};
  return s_rng;
}
#endif
