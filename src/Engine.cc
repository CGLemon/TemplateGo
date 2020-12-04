#include "Engine.h"
#include "Utils.h"
#include "Model.h"
#include "Network.h"
#include <cassert>

void Engine::initialize() {
    while (m_states.size() < (unsigned)option<int>("num_games")) {
        m_states.emplace_back(std::make_shared<GameState>());
    }

    if (m_states.size() > (unsigned)option<int>("num_games")) {
        m_states.resize((unsigned)option<int>("num_games"));
    }
    m_states.shrink_to_fit();

    for (auto & s: m_states) {
        s->init_game(option<int>("boardsize"), option<float>("komi"));
    }

    adjust_range(default_id, m_states.size());
    m_state = m_states[default_id];


    m_evaluation = std::make_shared<Evaluation>();
    m_evaluation->initialize_network(option<int>("playouts"), option<std::string>("weights_file"));

    m_trainer = std::make_shared<Trainer>();
    m_search = std::make_shared<Search>(*m_state, *m_evaluation, *m_trainer);
}

Engine::~Engine() {
    Utils::auto_printf("Engine was released\n");
}

void Engine::release() {
    m_trainer.reset();
    m_trainer = nullptr;

    m_search.reset();
    m_search = nullptr;

    m_evaluation->release_nn();
    m_evaluation.reset();
    m_evaluation = nullptr;
}

void Engine::display() {
    m_state->display(2);
}

Engine::Response Engine::play_textmove(std::string input) {
    m_state->play_textmove(input);
    return Response{};
}

Engine::Response Engine::showboard(int t) {
    return m_state->display_to_string(t);
}

Engine::Response Engine::nn_rawout(int symmetry) {

    auto res = m_evaluation->network_eval(*m_state,  Network::Ensemble::DIRECT, symmetry);
    auto pres = option<int>("float_precision");
    auto out = std::ostringstream{};

    out << "Winrate Misc :" << std::endl;
    out << " " <<  std::fixed << std::setprecision(pres) << res.alpha << std::endl;
    out << " " <<  std::fixed << std::setprecision(pres) << res.beta << std::endl;
    out << " " <<  std::fixed << std::setprecision(pres) << res.gamma << std::endl;

    const auto w = Model::get_winrate(*m_state, res);
    out << "Winrate :" << std::endl;
    out << " " <<  std::fixed << std::setprecision(pres) << w << std::endl;

    const auto bsize = m_state->get_boardsize();
    out << "Policy :" << std::endl;
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            auto idx = m_state->get_index(x, y);
            out << " " << std::setw(5) <<  std::fixed << std::setprecision(pres) << res.policy[idx];
        }
        out << std::endl;
    }
    out << "Pass :" << std::endl;
    out << " " << std::setw(5) << std::fixed << std::setprecision(pres) << res.policy_pass << std::endl;

    out << "Ownership :" << std::endl;
    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            auto idx = m_state->get_index(x, y);
            out << " " << std::setw(8) <<  std::fixed << std::setprecision(pres) << res.ownership[idx];
        }
        out << std::endl;
    }
    out << std::endl;
    out << "Fianl Score :" << std::endl;
    out << " " << std::setw(5) << std::fixed << std::setprecision(pres) << res.final_score << std::endl;

    return out.str();
}

Engine::Response Engine::reset_boardsize(const int bsize) {
    set_option("boardsize", bsize);
    auto komi = m_state->get_komi();
    m_state->init_game(bsize, komi);
    return Response{};
}

Engine::Response Engine::reset_komi(const float komi) {
    set_option("komi", komi);
    m_state->set_komi(komi);
    return Response{};
}

Engine::Response Engine::input_features(int symmetry) {
    if (symmetry < 0) {
        symmetry = 0;
    } else if (symmetry >= 8) {
        symmetry = 7;
    }
    return Model::features_to_string(*m_state, symmetry);
}

Engine::Response Engine::undo_move() {
    m_state->undo_move();
    if (m_state->get_komi() != option<float>("komi")) {
        m_state->set_komi(option<float>("komi"));
    }
    return Response{};
}

Engine::Response Engine::think(const int color) {
    if (color == Board::BLACK || color == Board::WHITE) {
        m_state->set_to_move(color);
    }
    auto move = m_search->think();
    m_state->play_move(move);
    return m_state->vertex_to_string(move);
}

Engine::Response Engine::self_play() {
    while(!m_state->isGameOver()) {
        auto move = m_search->think(Search::strategy_t::NN_UCT);
        m_state->play_move(move);

        auto res = m_state->vertex_to_string(move);
        auto_printf("move = %s\n", res.c_str());
        auto_printf("%s", showboard(2).c_str());
    }
    m_trainer->gather_winner(*m_state);
    return m_state->result_to_string();
}

Engine::Response Engine::dump_collect(std::string file) {
    auto out = std::ostringstream{};
    if (file == "std-output") {
        m_trainer->data_stream(out);
    } else {
        m_trainer->save_data(file, true);
        m_trainer->clear_game_steps();
    }
    return out.str();
}

Engine::Response Engine::dump_sgf(std::string file) {
    auto out = std::ostringstream{};
    if (file == "std-output") {
        SGFStream::sgf_stream(out, *m_state);
    } else {
        SGFStream::save_sgf(file, *m_state, true);
    }
    return out.str();
}

Engine::Response Engine::set_playouts(const int p) {
    m_evaluation->set_playouts(p);
    return Response{};
}

Engine::Response Engine::clear_board() {
    auto komi = m_state->get_komi();
    auto bsize = m_state->get_boardsize();
    m_state->init_game(bsize, komi);
    return Response{};
}

Engine::Response Engine::misc_features() {
    
    auto out = std::ostringstream{};
    const auto bsize = m_state->get_boardsize();

    out << "Ladder Plane:" << std::endl;
    out << " 0: None" << std::endl;
    out << " 1: Ladder is death." << std::endl;
    out << " 2: Ladder is escapable." << std::endl;
    out << " 3: Atari move." << std::endl;
    out << " 4: Take move." << std::endl;

    auto ladders = m_state->board.get_ladders();

    for (int y = 0; y < bsize; ++y) {
        for (int x = 0; x < bsize; ++x) {
            const auto idx = m_state->get_index(x, y);
            const auto ladder = ladders[idx];
            if (ladder == Board::ladder_t::LADDER_DEATH) {
                out << std::setw(4) << 1;
            } else if (ladder == Board::ladder_t::LADDER_ESCAPABLE) {
                out << std::setw(4) << 2;
            } else if (ladder == Board::ladder_t::LADDER_ATARI) {
                out << std::setw(4) << 3;
            } else if (ladder == Board::ladder_t::LADDER_TAKE) {
                out << std::setw(4) << 4;
            } else {
                assert(ladder == Board::ladder_t::NOT_LADDER);
                out << std::setw(4) << 0;
            }
        }
        out << std::endl;
    }
    return out.str();
}

Engine::Response Engine::nn_batchmark(const int times) {

    auto out = std::ostringstream{};
    const auto seconds = m_evaluation->nn_benchmark(*m_state, times);

    out << "Run times : " << times << std::endl;
    out << "Total : " << seconds << " second(s)" << " | ";
    out << "Avg : " << seconds / (float)times << " second(s)/time";
    return out.str();
}

Engine::Response Engine::clear_cache() {
    m_evaluation->clear_cache();
    return Response{};
}

Engine::Response Engine::random_playmove() {
    auto move = m_search->think(Search::strategy_t::RANDOM);
    m_state->play_move(move);
    return m_state->vertex_to_string(move);
}

Engine::Response Engine::time_settings(int maintime, int byotime, int byostones) {

    set_option("maintime", maintime);
    set_option("byotime", byotime);
    set_option("byostones", byostones);
    m_state->reset_time();
    return Response{};
}

Engine::Response Engine::time_left(std::string color, int time, int stones) {

    if (color == "b" || color == "B" || color == "black") {
        m_state->set_time_left(Board::BLACK, time, stones);
    } else if (color == "w" || color == "W" || color == "white") {
        m_state->set_time_left(Board::WHITE, time, stones);
    }

    return Response{};
}

const GameState& Engine::get_state() const {
    return *m_state;
}

