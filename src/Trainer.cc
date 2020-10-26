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

   /*
    * The each training data is consist of 9 different datas.
    * Thay are:
    *
    * 1. board size
    * 2. current player komi
    * 3. input binary planes
    * 4. input features
    * 5. current player probabilities
    * 6. next player probabilities
    * 7. current player ownership
    * 8. current player final score (without komi)
    * 9. winner
    *
    */
  
    const auto lambda_pow = [](const int base, const int exp) -> float {
        int val = 1;
        for (int i = 0; i < exp ; ++i) {
            val *= base;
        }
        return val;
    };

    const auto lambda_array_to_stream = [](const auto &array,
                                           std::ostream &out) -> void {
        auto b = std::begin(array);
        auto e = std::end(array);
        if (b != e) {
            e = std::prev(e);
        }
        if (b != e) {
            std::for_each(b, e, [&](auto in){ out << in << " "; });
        }
        out << *e << std::endl;
    };

    // 1. board size at this game.
    out << board_size << std::endl;

    // 2. current komi at this game.
    out << current_komi << std::endl;

    // 3. input binary planes
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

    // 4. input features
    lambda_array_to_stream(input_features, out);

    // 5. and 6. Probabilities
    lambda_array_to_stream(probabilities, out);
    lambda_array_to_stream(opponent_probabilities, out);

    // 7. Ownership
    lambda_array_to_stream(ownership, out);

    // 8. Final score
    out << final_score << std::endl;

    // 9. Winner
    out << result << std::endl;
}

bool gather_probabilities(GameState &state, UCTNode &node, std::vector<float> &probabilities, float temperature) {

    const size_t boardsize = state.board.get_boardsize();
    const size_t intersections = boardsize * boardsize;

    if (probabilities.size() != intersections+1) {
        return false;
    }

    auto factor = double{0.0f};
    auto tot_visits = size_t{0};

    node.inflate_all_children();
    const auto children = node.get_children();
  
    for (const auto &child : children) {
        const auto vertex = child->get()->get_vertex();
        const auto visits = child->get()->get_visits();
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

// Record the step from MCTS.
void Trainer::gather_step(GameState &state, UCTNode &node) {

    if (!option<bool>("collect")) {
        return;
    }

    const auto intersections = state.get_intersections();
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

// Record the step without any search.
void Trainer::gather_step(GameState &state, const int vtx) {

    if (!option<bool>("collect")) {
        return;
    }

    if (vtx == Board::RESIGN) {
        return;
    }

    const auto intersections = state.get_intersections();
    auto step = Step{};
  
    step.probabilities = std::vector<float>(intersections+1, 0.0f);
    int idx = Board::NO_INDEX;
    if (vtx == Board::PASS) {
        idx = intersections;
    } else {
        const auto x = state.get_x(vtx);
        const auto y = state.get_y(vtx);
        idx = state.get_index(x, y);
    }
    assert(idx != Board::NO_INDEX);

    step.probabilities[idx] = 1.0f;
    scatch_step(state, step);
    push_game_step(step);
}

void Trainer::scatch_step(GameState &state, Step &step) const {
    // input binary planes
    const auto planes = 
        Model::gather_planes(&state, Board::IDENTITY_SYMMETRY);

    step.input_planes = std::vector<char>(planes.size(), 0);
    for (auto idx = size_t{0}; idx < planes.size(); ++idx) {
        step.input_planes[idx] =
            static_cast<char>(planes[idx]);
    }

    // input features
    step.input_features = Model::gather_features(&state);

    // current player color
    step.to_move = state.board.get_to_move();

    // current game board size
    step.board_size = state.board.get_boardsize();

    // current komi.
    const auto komi = state.get_komi();
    if (komi == 0.0f) {
        step.current_komi = 0.0f;
    } else {
        step.current_komi = (step.to_move == Board::BLACK ? komi : -komi);
    }
}


void Trainer::gather_winner(GameState &state) {

    if (!option<bool>("collect")) {
        return;
    }

    const auto winner = state.get_winner();
    if (winner == Board::INVAL) {
        return;
    }

    const auto board_size = state.get_boardsize();
    const auto intersections = state.get_intersections();
    const auto ownership = state.board.get_ownership();
    const auto distance = state.board.area_distance();
    assert(winner != Board::INVAL);

    for (auto &step : game_steps) {
        assert(board_size == step.board_size);

        if (winner == Board::EMPTY) {
            step.result = 0.0f;
        } else {
            if (winner == step.to_move) {
                step.result = 1.0f;
            } else {
                step.result = -1.0f;
            }
        }

        assert(step.ownership.empty());
        step.ownership.reserve(ownership.size());

        for (auto &color : ownership) {
            assert(color != Board::INVAL);
            if (step.to_move == color) {
                step.ownership.emplace_back(1.0f);
            } else if ((!step.to_move) == color) {
                step.ownership.emplace_back(-1.0f);
            } else if (color == Board::EMPTY) {
                step.ownership.emplace_back(0.0f);
            }
        }
        assert(step.ownership.size() == ownership.size());
        step.final_score =
            (step.to_move == Board::BLACK) ? distance : (-distance);
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
        assert(ite->opponent_probabilities.size() == (size_t)(intersections+1));
    }
}

void Trainer::push_game_step(Step &step) {
    game_steps.emplace_back(std::move(step));
    adjust_game_steps(option<int>("max_game_buffer"));
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
