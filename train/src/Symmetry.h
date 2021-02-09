#ifndef SYMMETRT_H_INCLUDE
#define SYMMETRT_H_INCLUDE

#include <vector>
#include <array>

class Symmetry {
public:
    Symmetry() = delete;

    Symmetry(const size_t boardsize) {
        initialize(boardsize);
    }

    static constexpr size_t NUM_SYMMETRIES = 8;
    static constexpr size_t IDENTITY_SYMMETRY = 0;

    void initialize(const size_t boardsize);
    int get_index(const int x, const int y, const int boardsize) const;
    int get_transform_idx(const int idx, const int sym) const;

    void dump_symmetry_table() const;

private:
    std::pair<int, int> get_symmetry(const int x, const int y,
                                     const int symmetry,
                                     const int boardsize);

};

extern std::array<std::vector<int>, Symmetry::NUM_SYMMETRIES> symmetry_nn_idx_table;

#endif
