#include "Symmetry.h"
#include <cassert>

constexpr size_t Symmetry::NUM_SYMMETRIES;
constexpr size_t Symmetry::IDENTITY_SYMMETRY;
std::array<std::vector<int>, Symmetry::NUM_SYMMETRIES> symmetry_nn_idx_table;

void Symmetry::initialize(const size_t boardsize) {


    const size_t intersections = boardsize * boardsize;

    for (auto sym = size_t{0}; sym < NUM_SYMMETRIES; ++sym) {
        symmetry_nn_idx_table[sym].clear();
        symmetry_nn_idx_table[sym] = std::vector<int>(intersections);
        for (auto idx = size_t{0}; idx < intersections; ++idx) {
            symmetry_nn_idx_table[sym][idx] = 0;
        }
    }


    for (auto sym = size_t{0}; sym < NUM_SYMMETRIES; ++sym) {
        for (auto y = size_t{0}; y < boardsize; y++) {
            for (auto x = size_t{0}; x < boardsize; x++) {
                const auto sym_idx = get_symmetry(x, y, sym, boardsize);
                const auto idx = get_index(x, y, boardsize);
                symmetry_nn_idx_table[sym][idx] = 
                    get_index(sym_idx.first, sym_idx.second, boardsize);
            }
        }
    }
}

int Symmetry::get_index(const int x, const int y, const int boardsize) const {
      return y * boardsize + x;
}

std::pair<int, int> Symmetry::get_symmetry(const int x, const int y,
                                           const int symmetry,
                                           const int boardsize) {

    assert(x >= 0 && x < boardsize);
    assert(y >= 0 && y < boardsize);
    assert(symmetry >= 0 && symmetry < (int)NUM_SYMMETRIES);

    constexpr std::uint32_t X_AXIS_REFLECT = 1;
    constexpr std::uint32_t Y_AXIS_REFLECT = 1 << 1;
    constexpr std::uint32_t DIAGONAL       = 1 << 2;

    const std::uint32_t Symmetry__ = static_cast<std::uint32_t>(symmetry);

    int idx_x = x;
    int idx_y = y;

    if ((Symmetry__ & DIAGONAL) != 0) {
        std::swap(idx_x, idx_y);
    }

    if ((Symmetry__ & Y_AXIS_REFLECT) != 0) {
        idx_x = boardsize - idx_x - 1;
    }

    if ((Symmetry__ & X_AXIS_REFLECT) != 0) {
        idx_y = boardsize - idx_y - 1;
    }

    assert(idx_x >= 0 && idx_x < boardsize);
    assert(idx_y >= 0 && idx_y < boardsize);
    assert(symmetry != IDENTITY_SYMMETRY || (x == idx_x && y == idx_y));

    return std::pair<int, int>{idx_x, idx_y};
}

int Symmetry::get_transform_idx(const int idx, const int sym) const {
    return symmetry_nn_idx_table[sym][idx];
}

void Symmetry::dump_symmetry_table() const {

    const auto lambda_sqrt = [](size_t m){
        auto res = size_t{0};
        for (auto factor = size_t{0}; factor * factor <= m; ++factor) {
            if (factor * factor == m) {
                res = factor;
                break;
            }
        }
        return res;
    };

    printf("================= Table =================\n");
    const auto lengh = symmetry_nn_idx_table[IDENTITY_SYMMETRY].size();
    const auto bsize = lambda_sqrt(lengh);
    assert(bsize != 0);
    for (auto sym = size_t{0}; sym < NUM_SYMMETRIES; ++sym) {
        if (sym == 0) {
            printf("Original :\n");
        }

        for (auto y = size_t{0}; y < bsize; y++) {
            for (auto x = size_t{0}; x < bsize; x++) {
                const auto idx = get_index(x, y, bsize);
                printf("%3d ", symmetry_nn_idx_table[sym][idx]);
            }
            printf("\n");
        }
        printf("\n");
    }
    printf("=========================================\n");
}

