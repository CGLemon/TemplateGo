#include <algorithm>
#include <cassert>

#include "Random.h"
#include "Zobrist.h"
#include "config.h"

constexpr Zobrist::KEY Zobrist::zobrist_seed;
constexpr Zobrist::KEY Zobrist::zobrist_empty;
constexpr Zobrist::KEY Zobrist::zobrist_blacktomove;

std::array<std::array<Zobrist::KEY, Zobrist::ZOBRIST_SIZE>, 4> Zobrist::zobrist;
std::array<std::array<Zobrist::KEY, Zobrist::ZOBRIST_SIZE * 2>, 2> Zobrist::zobrist_pris;

std::array<Zobrist::KEY, Zobrist::ZOBRIST_SIZE> Zobrist::zobrist_ko;
std::array<Zobrist::KEY, 5> Zobrist::zobrist_pass;

template<typename T>
bool collision(std::vector<T> &array) {
    const auto s = array.size();
    if (s <= 1) {
        return false;
    }

    for (auto i = size_t{0}; i < (s-1); ++i) {
        auto begin = std::cbegin(array);
        auto element = std::next(begin, i);
        auto start = std::next(element, 1);
        auto end = std::cend(array);
        auto res = std::find(start, end, *element);
        if (res != end) {
            return true;
        }
    }
    return false;
}

void Zobrist::init_zobrist() {

    Random<random_t::XoroShiro128Plus> rng(zobrist_seed);

    while (true) {
        auto buf = std::vector<KEY>{};

        for (int i = 0; i < 4; ++i) {
            for (int j = 0; j < ZOBRIST_SIZE; ++j) {
                Zobrist::zobrist[i][j] = rng.randuint64();
                buf.emplace_back(Zobrist::zobrist[i][j]);
            }
        }

        for (int i = 0; i < 2; ++i) {
            for (int j = 0; j < ZOBRIST_SIZE * 2; ++j) {
                Zobrist::zobrist_pris[i][j] = rng.randuint64();
                buf.emplace_back(Zobrist::zobrist_pris[i][j]);
            }
        }

        for (int i = 0; i < ZOBRIST_SIZE; ++i) {
            Zobrist::zobrist_ko[i] = rng.randuint64();
            buf.emplace_back(Zobrist::zobrist_ko[i]);
        }

        for (int i = 0; i < 5; ++i) {
            Zobrist::zobrist_pass[i] = rng.randuint64();
            buf.emplace_back(Zobrist::zobrist_pass[i]);
        }

        if (!collision(buf)) {
            break;
        }
    }
}
