#ifndef EVALUATION_H_INCLUDE
#define EVALUATION_H_INCLUDE

#include <array>
#include <vector>
#include <iomanip>

#include "Board.h"
#include "GameState.h"
#include "Network.h"
#include "config.h"
#include "Model.h"

class Evaluation {
public:
    using NNeval = NNResult;

    void initialize_network(int playouts, const std::string &weightsfile);
    NNeval network_eval(GameState &state,
                        Network::Ensemble ensemble = Network::RANDOM_SYMMETRY);

    void reload_network(std::string &weightsfile);

    void clear_cache();

    void release_nn();

    void set_playouts(const int p);

    float nn_benchmark(GameState &state, const int times);

private:
    Network m_network;

};

#endif
