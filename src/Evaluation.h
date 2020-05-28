#ifndef EVALUATION_H_INCLUDE
#define EVALUATION_H_INCLUDE

#include <vector>
#include <array>

#include "config.h"
#include "Board.h"
#include "GameState.h"
#include "Network.h"
#include "CacheTable.h"

class Evaluation {
public:
	using NNeval = NNResult;

	void initialize_network(int playouts, const std::string & weightsfile);
	NNeval network_eval(GameState& state, Network::Ensemble ensemble = Network::RANDOM_SYMMETRY);

private:
	Network m_network;
};




#endif
