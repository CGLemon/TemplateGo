#ifndef EVALUATION_H_INCLUDE
#define EVALUATION_H_INCLUDE

#include <array>
#include <vector>

#include "Board.h"
#include "CacheTable.h"
#include "GameState.h"
#include "Network.h"
#include "config.h"

class Evaluation {
public:
  using NNeval = NNResult;

  void initialize_network(int playouts, const std::string &weightsfile);
  NNeval network_eval(GameState &state,
                      Network::Ensemble ensemble = Network::RANDOM_SYMMETRY);

private:
  Network m_network;
};

#endif
