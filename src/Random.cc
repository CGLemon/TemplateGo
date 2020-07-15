#include "Random.h"
using namespace random_utils;

template<random_t T>
thread_local std::array<std::uint64_t,
                 Random<T>::SEED_SZIE> Random<T>::m_seeds;


template<>
std::uint64_t Random<random_t::SplitMix_64>::randuint64() {
  m_seeds[0] += 0x9e3779b97f4a7c15;
  auto z = m_seeds[0];
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
  z = (z ^ (z >> 27)) * 0x94d049bb133111eb;
  return z ^ (z >> 31);
}

template<>
std::uint64_t Random<random_t::XorShiro128Plus>::randuint64() {
  /*
  The parameter detail are from
  https://github.com/lemire/testingRNG/blob/master/source/Random.h
  */

  const std::uint64_t s0 = m_seeds[0];
  std::uint64_t s1 = m_seeds[1];
  const std::uint64_t result = s0 + s1;

  s1 ^= s0;
  m_seeds[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
  m_seeds[1] = rotl(s1, 36);

  return result;
}
