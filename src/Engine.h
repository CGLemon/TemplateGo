#ifndef ENGINE_H_INCLUDE
#define ENGINE_H_INCLUDE

#include "Evaluation.h"
#include "Search.h"
#include "Trainer.h"
#include "config.h"
#include "GameState.h"

#include <memory>

struct Engine {
  Engine(GameState &state) : m_state(state) {}

  GameState &m_state;

  std::shared_ptr<Evaluation> evaluation;
  std::shared_ptr<Search> search;
  std::shared_ptr<Trainer> trainer;

  void init();

  GameState& get_state();

  float get_fair_komi();

  int think(Search::strategy_t stg);

  void benchmark(int playouts);

  void clear_state();
};

#endif
