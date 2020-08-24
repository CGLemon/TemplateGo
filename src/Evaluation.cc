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

int Evaluation::get_fair_komi(GameState &state) {
  const auto out = network_eval(state, Network::Ensemble::NONE);
  const auto intersections = state.board.get_intersections();

  float max_score_prob = std::numeric_limits<float>::lowest();
  int max_final_score = std::numeric_limits<int>::lowest();
  for (auto idx = size_t{0}; idx < 2 * intersections; ++idx) {
    if (max_score_prob < out.final_score[idx]) {
      max_score_prob = out.final_score[idx];
      max_final_score = static_cast<int>(idx);
    }
  }

  auto fair_komi = max_final_score - intersections;
  if (state.board.get_to_move() == Board::WHITE) {
    fair_komi = 0 - fair_komi;
  }

  return fair_komi;
}
