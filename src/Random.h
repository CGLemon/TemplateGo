#ifndef RANDON_H_INCLUDE
#define RANDON_H_INCLUDE


#include <cstdint>
#include <limits>
#include <time.h>
#include <random>
#include <thread>

/*
 https://github.com/lemire/testingRNG
 */

namespace random_utils {
static inline std::uint64_t splitmix64(std::uint64_t z) {
    /*
     The parameter detail are from
     https://github.com/lemire/testingRNG/blob/master/source/splitmix64.h
     */

    z += 0x9e3779b97f4a7c15;
    z = (z^(z >> 30)) * 0xbf58476d1ce4e5b9;
    z = (z^(z >> 27)) * 0x94d049bb133111eb;
    return z^(z >> 31);
}

static inline std::uint64_t rotl(const std::uint64_t x, const int k) {
  	return (x << k) | (x >> (64 - k));
}

}


class Random {    
public:
    
    Random() = delete;
    Random(std::uint64_t default_seed = 0);
    
    static Random& get_Rng(const std::uint64_t seed = 0);
    
    void seed_init(std::uint64_t);
    
    std::uint64_t randuint64() {
        return xoroshiro128plus();
    }
    
	void print_seed();
    
	template<int MAX>
    std::uint32_t randfix() {
        static_assert(0 < MAX &&
                     MAX < std::numeric_limits<std::uint32_t>::max(),
                     "randfix out of range");
        // Last bit isn't random, so don't use it in isolation. We specialize
        // this case.
        static_assert(MAX != 2, "don't isolate the LSB with xoroshiro128+");
        return xoroshiro128plus() % MAX;
    }



    using result_type = std::uint64_t;
    constexpr static result_type min() {
        return std::numeric_limits<result_type>::min();
    }
    constexpr static result_type max() {
        return std::numeric_limits<result_type>::max();
    }
    result_type operator()() {
        return xoroshiro128plus();
    }
    
private:
    std::uint64_t xoroshiro128plus();
    static thread_local std::uint64_t m_s[2];
};

#endif
