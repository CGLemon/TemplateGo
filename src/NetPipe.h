#ifndef NETPIPE_H_INCLUDED
#define NETPIPE_H_INCLUDED

#include <memory>
#include <vector>

#include "config.h"

static constexpr int WINOGRAD_M = 4;
static constexpr int WINOGRAD_ALPHA = WINOGRAD_M + 3 - 1;
static constexpr int WINOGRAD_WTILES = BOARD_SIZE / WINOGRAD_M + (BOARD_SIZE % WINOGRAD_M != 0);
static constexpr int WINOGRAD_TILE = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
static constexpr int WINOGRAD_P = WINOGRAD_WTILES * WINOGRAD_WTILES;
static constexpr float SQ2 = 1.4142135623730951f; // Square root of 2

static constexpr int OUTPUTS_POLICY = 2;
static constexpr int OUTPUTS_VALUE = 1;
static constexpr int INPUT_MOVES = 8;
static constexpr int INPUT_CHANNELS = 2 * INPUT_MOVES + 2;
static constexpr int VALUE_LAYER = 256;

static constexpr int VALUE_LABELS = 1;
static constexpr int POTENTIAL_MOVES = NUM_INTERSECTIONS+1;

static constexpr int CONV2D_SIZE = BOARD_SIZE;


struct ForwardPipeWeights {
    // Input + residual block tower
    std::vector<std::vector<float>> m_conv_weights;
    std::vector<std::vector<float>> m_conv_biases;
    std::vector<std::vector<float>> m_batchnorm_means;
    std::vector<std::vector<float>> m_batchnorm_stddevs;

    // Policy head
    std::vector<float> m_conv_pol_w;
    std::vector<float> m_conv_pol_b;

    std::vector<float> m_conv_val_w;
    std::vector<float> m_conv_val_b;
};

class ForwardPipe {
public:
    
    virtual ~ForwardPipe() = default;

    virtual void initialize(const int channels) = 0;
    virtual bool needs_autodetect() { return false; };
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val) = 0;
    virtual void push_weights(unsigned int filter_size,
                              unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights) = 0;
};


#endif
