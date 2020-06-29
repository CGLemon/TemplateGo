#ifndef LZNETPARAMETER_H_INCLUDE
#define LZNETPARAMETER_H_INCLUDE

#include "config.h"
namespace LZ {

static constexpr int OUTPUTS_POLICY = 2;
static constexpr int OUTPUTS_VALUE = 1;
static constexpr int INPUT_MOVES = 8;
static constexpr int INPUT_CHANNELS = 2 * INPUT_MOVES + 2;
static constexpr int VALUE_LAYER = 256;

static constexpr int VALUE_LABELS = 1;
static constexpr int POTENTIAL_MOVES = NUM_INTERSECTIONS + 1;
static constexpr int RESIDUAL_FILTER = 3;
static constexpr int HEAD_FILTER = 1;

}

#endif
