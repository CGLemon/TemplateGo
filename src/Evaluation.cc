#include "Evaluation.h"
#include "Board.h"
#include "GameState.h"
#include "Random.h"
#include "Network.h"

#include <memory>
#include <cstdint>
#include <cassert>

void Evaluation::initialize_network(int playouts, const std::string & weightsfile) {
	m_network.initialize(playouts, weightsfile);
}


Evaluation::NNeval Evaluation::network_eval(GameState& state, Network::Ensemble ensemble) {
	return m_network.get_output(&state, Network::Ensemble::RANDOM_SYMMETRY);
}
