#include "Engine.h"
#include <cassert>

void Engine::init() {
  evaluation = std::make_shared<Evaluation>();
  evaluation->initialize_network(cfg_playouts, cfg_weightsfile);
  trainer=std::make_shared<Trainer>();
  search = std::make_shared<Search>(m_state, *evaluation, *trainer);
}

int Engine::think(Search::strategy_t stg) {
  return search->think(stg);
}

void Engine::benchmark(int playouts) {
  search->benchmark(playouts);
}

void Engine::clear_state() {
  const int boardsize = m_state.board.get_boardsize();
  const float komi = m_state.board.get_komi();

  assert(komi == cfg_komi && boardsize == cfg_boardsize);

  m_state.init_game(cfg_boardsize, cfg_komi);
  evaluation->clear_cache();
}

float Engine::get_fair_komi() {
  return evaluation->get_fair_komi(m_state);
}

GameState& Engine::get_state() {
  return m_state;
}
