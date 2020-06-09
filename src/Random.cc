#include "Random.h"
using namespace random_utils;

thread_local std::uint64_t Random::m_s[2];

Random &Random::get_Rng(const std::uint64_t seed) {
  static thread_local Random s_rng{seed};
  return s_rng;
}

Random::Random(std::uint64_t seed) {
  if (seed == 0) {
    size_t thread_id = std::hash<std::thread::id>()(std::this_thread::get_id());
    seed_init(std::uint64_t(thread_id));
  } else {
    seed_init(seed);
  }
}

void Random::seed_init(std::uint64_t default_seed) {
  m_s[0] = splitmix64(default_seed);
  m_s[1] = splitmix64(m_s[0]);
}

std::uint64_t Random::xoroshiro128plus() {
  /*
  The parameter detail are from
  https://github.com/lemire/testingRNG/blob/master/source/Random.h
  */

  const std::uint64_t s0 = m_s[0];
  std::uint64_t s1 = m_s[1];
  const uint64_t result = s0 + s1;

  s1 ^= s0;
  m_s[0] = rotl(s0, 55) ^ s1 ^ (s1 << 14);
  m_s[1] = rotl(s1, 36);

  return result;
}
