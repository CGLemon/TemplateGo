#include "Trainer.h"
#include "Model.h"
#include "Board.h"
#include "config.h"
#include "Utils.h"
#include "Random.h"

#include <algorithm>
#include <memory>
#include <string>
#include <fstream>
#include <iostream>
#include <sstream>

using namespace Utils;

void Trainer::Step::step_stream(std::ostream &out) {
  
  const auto lambda_pow = [](const int base, const int exp){
    int val = 1;
    for (int i = 0; i < exp ; ++i) {
      val *= base;
    }
    return val;
  };

  out << board_size << std::endl;

  const auto input_planes_size = input_planes.size(); 
  auto idx = size_t{0};
  while (idx < input_planes_size) {
    auto hex = 0;

    for (auto i = size_t{0}; i < 4; ++i) {
      hex += input_planes[idx] * lambda_pow(2, (3 - i));
      idx++;
      if (idx >= input_planes_size) {
        break;
      }
    }
    out << std::hex << hex;
  }
  assert(idx == input_planes_size);
  out << std::dec << std::endl;

  for (auto &in : input_features) {
    out << in << " ";
  }
  out << std::endl;

  for (auto &p : probabilities) {
    out << p << " ";
  }
  out << std::endl;

  for (auto &p : opponent_probabilities) {
    out << p << " ";
  }

  out << std::endl;

  for (auto &o : ownership) {
    out << o << " ";
  }
  out << std::endl;

  out << final_score_idx;
  out << std::endl;

  for (auto &r : result) {
    out << r << " ";
  }
  out << std::endl;
}

bool gather_probabilities(GameState &state, UCTNode &node, std::vector<float> &probabilities, float temperature) {

  const size_t boardsize = state.board.get_boardsize();
  const size_t intersections = boardsize * boardsize;

  if (probabilities.size() != intersections+1) {
    return false;
  }


  auto factor = double{0.0f};
  auto tot_visits = size_t{0};
  const auto children = node.get_children();

  for (const auto &child : children) {
    const auto vertex = child->get_vertex();
    const auto visits = child->get_visits();
    int idx = Board::NO_INDEX;
    if (vertex == Board::PASS) {
      idx = intersections;
    } else {
      const auto x = state.board.get_x(vertex);
      const auto y = state.board.get_y(vertex);
      idx = state.board.get_index(x, y);
    }

    if (visits > 1) {
      const double exponent = 1.0f / temperature;
      const double visits_with_temperature = 
                       std::pow(static_cast<double>(visits), exponent);

      tot_visits += visits;
      factor += visits_with_temperature;
      probabilities[idx] = visits_with_temperature;
    } else {
      tot_visits += 0;
      factor += 0;
      probabilities[idx] = 0;
    }

    assert(idx != Board::NO_INDEX);
  }
  if (tot_visits == 0) {
    return false;
  }
  
  for (auto &p : probabilities) {
    p /= factor;
  }
  
  return true;
}

void Trainer::gather_step(GameState &state, UCTNode &node) {

  if (!cfg_collect) {
    return;
  }

  const size_t boardsize = state.board.get_boardsize();
  const size_t intersections = boardsize * boardsize;
  auto step = Step{};
  auto to_move = state.board.get_to_move();
  assert(to_move == node.get_color());

  step.probabilities = std::vector<float>(intersections+1, 0.0f);
  bool success = gather_probabilities(state, node, step.probabilities, 1.0f);

  if (!success) {
    return;
  }

  scatch_step(state, step);

  push_game_step(step);
}

void Trainer::gather_step(GameState &state, const int vtx) {

  if (!cfg_collect) {
    return;
  }

  // Record step without any search. 
  if (vtx == Board::RESIGN) {
    return;
  }

  const size_t boardsize = state.board.get_boardsize();
  const size_t intersections = boardsize * boardsize;
  auto step = Step{};
  
  step.probabilities = std::vector<float>(intersections+1, 0.0f);
  int idx = Board::NO_INDEX;
  if (vtx == Board::PASS) {
    idx = intersections;
  } else {
    const auto x = state.board.get_x(vtx);
    const auto y = state.board.get_y(vtx);
    idx = state.board.get_index(x, y);
  }
  assert(idx != Board::NO_INDEX);

  step.probabilities[idx] = 1.0f;

  scatch_step(state, step);

  push_game_step(step);
}

void Trainer::scatch_step(GameState &state, Step &step) const {
  const auto planes = 
      Model::gather_planes(&state, Board::IDENTITY_SYMMETRY);

  step.input_planes = std::vector<int>(planes.size(), 0);
  for (auto idx = size_t{0}; idx < planes.size(); ++idx) {
    step.input_planes[idx] =
        static_cast<char>(planes[idx]);
  }

  step.input_features = Model::gather_features(&state);
  step.to_move = state.board.get_to_move();
  step.board_size = state.board.get_boardsize();
}


void Trainer::gather_winner(GameState &state) {

  if (!cfg_collect) {
    return;
  }

  const auto winner = state.get_winner();
  if (winner == Board::INVAL) {
    return;
  }
  const auto board_size = state.board.get_boardsize();
  const auto intersections = state.board.get_intersections();
  const auto ownership = state.board.get_ownership();
  const auto distance = state.board.area_distance();
 
  
  auto ml_result = std::vector<int>{};

  for (int i = 0; i < 21; ++i) {
    const auto addtion_komi = i - 10;
    const auto board_score = state.final_score((float)addtion_komi);
    auto ml_winner = Board::INVAL;
    if (board_score == 0.0f) {
      ml_winner = Board::EMPTY;
    } else if (board_score < 0.0f) {
      ml_winner = Board::WHITE;
    } else if (board_score > 0.0f) {
      ml_winner = Board::BLACK;
    }
    assert(ml_winner != Board::INVAL);
    ml_result.emplace_back(ml_winner);
  } 


  for (auto &step : game_steps) {
    assert(board_size == step.board_size);

    for(auto &ml_winner : ml_result) {
      if (ml_winner == Board::EMPTY) {
        step.result.emplace_back(0.0f);
      } else {
        if (step.to_move == ml_winner) {
          step.result.emplace_back(1.0f);
        } else {
          step.result.emplace_back(-1.0f);
        }
      }
    }

    assert(step.ownership.empty());
    step.ownership.reserve(ownership.size());

    for (auto &color : ownership) {
      assert(color != Board::INVAL);
      if (step.to_move == color) {
        step.ownership.emplace_back(1);
      } else if ((!step.to_move) == color) {
        step.ownership.emplace_back(-1);
      } else if (color == Board::EMPTY) {
        step.ownership.emplace_back(0);
      }
    }
    assert(step.ownership.size() == ownership.size());

    auto final_score_idx = 2 * (distance - state.board.get_komi_integer());
    if (state.board.get_komi_float() > 0.0f) {
      final_score_idx += 1;
    }

    step.final_score_idx =
        (step.to_move == Board::BLACK) ? final_score_idx : (-final_score_idx);
  }

  const auto end = std::end(game_steps);

  for (auto ite = std::begin(game_steps); ite != end; ++ite) {
    auto next = ite;
    next++;
    if (next != end) {
      ite->opponent_probabilities = next->probabilities;
    } else {
      ite->opponent_probabilities = std::vector<float>(intersections+1, 0.0f);
      ite->opponent_probabilities[intersections] = 1.0f;
    }
    assert(ite->opponent_probabilities.size() == intersections+1);
  }
}

void Trainer::push_game_step(Step &step) {
  game_steps.emplace_back(std::move(step));
  adjust_game_steps(cfg_max_game_buffer);
}

void Trainer::adjust_game_steps(size_t size) {
  while (game_steps.size() > size) {
    game_steps.pop_front();
  }
}

void Trainer::clear_game_steps() {
  game_steps.clear();
}

void Trainer::dump_memory() const {

  size_t step_memroy_word = 0;
  const size_t buffer_size = game_steps.size();
  
  for (auto &x : game_steps) {
    step_memroy_word += sizeof(x);
  }

  const float memory_used = 
      static_cast<float>(buffer_size * step_memroy_word) / (1024.f * 1024.f);

  auto_printf("stores %zu steps\n", buffer_size);
  auto_printf("memory use %.5f (Mib)\n", memory_used);
}

void Trainer::data_stream(std::ostream &out) {
  for (auto &x : game_steps) {
    x.step_stream(out);
  }
}

void Trainer::save_data(std::string &filename, bool append) {

  auto ios_tag = std::ios::out;
  auto out = std::ostringstream{};
  data_stream(out);

  if (append) {
    ios_tag |= std::ios::app;
  }

  std::fstream save_file;
  
  save_file.open(filename, ios_tag);
  save_file << out.str();
  save_file.close();
}
