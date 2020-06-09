#include <algorithm>
#include <cassert>

#include "Random.h"
#include "Zobrist.h"
#include "config.h"

constexpr std::uint64_t Zobrist::zobrist_seed;
constexpr std::uint64_t Zobrist::zobrist_empty;
constexpr std::uint64_t Zobrist::zobrist_blacktomove;

std::array<std::array<std::uint64_t, NUM_VERTICES>, 4> Zobrist::zobrist;
std::array<std::uint64_t, NUM_VERTICES> Zobrist::zobrist_ko;
std::array<std::array<std::uint64_t, NUM_VERTICES * 2>, 2>
    Zobrist::zobrist_pris;
std::array<std::uint64_t, 5> Zobrist::zobrist_pass;

void Zobrist::init_zobrist() {

  auto rng = Random::get_Rng(zobrist_seed);

  auto count = size_t{0};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < NUM_VERTICES; j++) {
      Zobrist::zobrist[i][j] = rng.randuint64();
      count++;
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < NUM_VERTICES * 2; j++) {
      Zobrist::zobrist_pris[i][j] = rng.randuint64();
      count++;
    }
  }

  for (int i = 0; i < NUM_VERTICES; i++) {
    Zobrist::zobrist_ko[i] = rng.randuint64();
    count++;
  }

  for (int i = 0; i < 5; i++) {
    Zobrist::zobrist_pass[i] = rng.randuint64();
    count++;
  }

  assert(count == (NUM_VERTICES * 4) + (NUM_VERTICES * 4) + NUM_VERTICES + 5);
}
