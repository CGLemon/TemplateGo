#include <algorithm>
#include <cassert>

#include "Random.h"
#include "Zobrist.h"
#include "config.h"

constexpr std::uint64_t Zobrist::zobrist_seed;
constexpr std::uint64_t Zobrist::zobrist_empty;
constexpr std::uint64_t Zobrist::zobrist_blacktomove;

std::array<std::array<std::uint64_t, Zobrist::ZOBRIST_SIZE>, 4> Zobrist::zobrist;
std::array<std::uint64_t, Zobrist::ZOBRIST_SIZE> Zobrist::zobrist_ko;
std::array<std::array<std::uint64_t, Zobrist::ZOBRIST_SIZE * 2>, 2>
    Zobrist::zobrist_pris;
std::array<std::uint64_t, 5> Zobrist::zobrist_pass;

template<typename T>
bool is_same(std::vector<T> &array, T element) {
  auto begin = std::begin(array);
  auto end = std::end(array);
  auto res = std::find(begin, end, element);
  return (res != end);
} 

void Zobrist::init_zobrist() {

  Random<random_t::XorShiro128Plus> rng(zobrist_seed);

  auto buf = std::vector<std::uint64_t>{};
  auto count = size_t{0};

  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < ZOBRIST_SIZE; j++) {
      Zobrist::zobrist[i][j] = rng.randuint64();
      buf.emplace_back(Zobrist::zobrist[i][j]);
      count++;
    }
  }

  for (int i = 0; i < 2; i++) {
    for (int j = 0; j < ZOBRIST_SIZE * 2; j++) {
      Zobrist::zobrist_pris[i][j] = rng.randuint64();
      buf.emplace_back(Zobrist::zobrist_pris[i][j]);
      count++;
    }
  }

  for (int i = 0; i < ZOBRIST_SIZE; i++) {
    Zobrist::zobrist_ko[i] = rng.randuint64();
    buf.emplace_back(Zobrist::zobrist_ko[i]);
    count++;
  }

  for (int i = 0; i < 5; i++) {
    Zobrist::zobrist_pass[i] = rng.randuint64();
    buf.emplace_back(Zobrist::zobrist_pass[i]);
    count++;
  }

  assert(buf.size() == count);
  assert(count == (ZOBRIST_SIZE * 4) + (ZOBRIST_SIZE * 4) + ZOBRIST_SIZE + 5);

  for (auto &element : buf) {
    assert(is_same<std::uint64_t>(buf, element));
  }
}
