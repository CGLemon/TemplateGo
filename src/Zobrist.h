#ifndef ZOBRIST_H_INCLUDE
#define ZOBRIST_H_INCLUDE

#include <array>
#include <random>
#include <vector>

#include "config.h"

class Zobrist {
public:
    using KEY = std::uint64_t;

    static constexpr auto ZOBRIST_SIZE = NUM_VERTICES;

    static constexpr KEY zobrist_empty = 0x1234567887654321;

    static constexpr KEY zobrist_blacktomove = 0xabcdabcdabcdabcd;

    static std::array<std::array<KEY, ZOBRIST_SIZE>, 4> zobrist;

    static std::array<KEY, ZOBRIST_SIZE> zobrist_ko;

    static std::array<std::array<KEY, ZOBRIST_SIZE * 2>, 2> zobrist_pris;

    static std::array<KEY, 5> zobrist_pass;

    static void init_zobrist();

private:
    static constexpr KEY zobrist_seed = 0xabcdabcd12345678;

};

#endif
