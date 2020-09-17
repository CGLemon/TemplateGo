#include "Evaluation.h"
#include "Board.h"
#include "GameState.h"
#include "Network.h"
#include "Random.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>

void Evaluation::initialize_network(int playouts,
                                    const std::string &weightsfile) {
  m_network.initialize(playouts, weightsfile);
}

Evaluation::NNeval Evaluation::network_eval(GameState &state,
                                            Network::Ensemble ensemble) {
  return m_network.get_output(&state, ensemble);
}

void Evaluation::reload_network(std::string &weightsfile) {
  m_network.reload_weights(weightsfile);
}

void Evaluation::clear_cache() {
  m_network.clear_cache();
}

void Evaluation::release_nn() {
  m_network.release_nn();
}

float Evaluation::get_fair_komi(GameState &state) {
  return state.board.get_komi();
}
