#ifndef ZOBRIST_H_INCLUDE
#define ZOBRIST_H_INCLUDE

#include <array>
#include <random>
#include <vector>

#include "config.h"

class Zobrist {
public:
  static constexpr int ZOBRIST_SIZE = NUM_VERTICES;

  static constexpr std::uint64_t zobrist_empty = 0x1234567887654321;
  static constexpr std::uint64_t zobrist_blacktomove = 0xabcdabcdabcdabcd;

  static std::array<std::array<std::uint64_t, ZOBRIST_SIZE>, 4> zobrist;
  static std::array<std::uint64_t, ZOBRIST_SIZE> zobrist_ko;
  static std::array<std::array<std::uint64_t, ZOBRIST_SIZE * 2>, 2>
      zobrist_pris;
  static std::array<std::uint64_t, 5> zobrist_pass;

  static void init_zobrist();

private:
  static constexpr std::uint64_t zobrist_seed = 0xabcdabcd12345678;
};

#endif
