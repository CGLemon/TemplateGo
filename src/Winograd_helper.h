#ifndef WINOGRAD_HELPER_H_INCLUDE
#define WINOGRAD_HELPER_H_INCLUDE
#include <vector>
#include <array>
#include "config.h"

static constexpr int WINOGRAD_M = 4;
static constexpr int WINOGRAD_ALPHA = WINOGRAD_M + 3 - 1;
//static constexpr int WINOGRAD_WTILES =
//    BOARD_SIZE / WINOGRAD_M + (BOARD_SIZE % WINOGRAD_M != 0);
static constexpr int WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
//static constexpr int WINOGRAD_P = WINOGRAD_WTILES * WINOGRAD_WTILES;
static constexpr float SQ2 = 1.4142135623730951f; // Square root of 2


std::vector<float> winograd_transform_f(const std::vector<float> &f,
                                        const int outputs, const int channels);

#endif

