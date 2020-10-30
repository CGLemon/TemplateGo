#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <sstream>

#include "GameState.h"
#include "Utils.h"

void GameState::init_game(int size, float komi) {

    board.reset_board(size, komi);

    reset_time();

    m_game_history.clear();
    m_game_history.emplace_back(std::make_shared<Board>(board));

    m_kohash_history.clear();
    m_kohash_history.emplace_back(board.get_ko_hash());

    m_resigned = Board::INVAL;
}

bool GameState::play_move(const int vtx, const int color) {

    if (isGameOver()) {
      return false;
    }

    if (!board.is_legal(vtx, color)) {
        return false;
    }

    if (option<bool>("pre_block_superko")) {
        if (board.is_superko_move(vtx, color, m_kohash_history)) {
            return false;
        }
    }

    if (vtx == Board::RESIGN) {
        set_to_move(color);
        m_resigned = color;
        return true;
    } else {
        board.play_move(vtx, color);
    }

    const auto movenum = board.get_movenum();
    assert((unsigned)movenum == m_game_history.size());

    m_game_history.emplace_back(std::make_shared<Board>(board));
    m_game_history.resize(movenum + 1);

    m_kohash_history.emplace_back(board.get_ko_hash());
    m_kohash_history.resize(movenum + 1);

    return true;
}

bool GameState::undo_move() {

    const auto movenum = board.get_movenum();
    if (movenum > 0) {
        board = *m_game_history[movenum - 1];
        m_game_history.resize(movenum);
        m_kohash_history.resize(movenum);
        return true;
    }
    return false;
}

bool GameState::play_textmove(std::string input) {

    int color = Board::INVAL;
    int vertex = Board::NO_VERTEX;

    const auto text2vertex = [this](const std::string &text) -> int {
        if (text.size() < 2) {
            return Board::NO_VERTEX;
        }

        if (text == "PASS" || text == "pass") {
            return Board::PASS;
        } else if (text == "RESIGN" || text == "resign") {
            return Board::RESIGN;
        }

        int x = -1;
        int y = -1;
        if (text[0] >= 'a' && text[0] <= 'z') {
            x = text[0] - 'a';
            if (text[0] >= 'i')
                x--;
        } else if (text[0] >= 'A' && text[0] <= 'Z') {
            x = text[0] - 'A';
            if (text[0] >= 'I')
                x--;
        }
        auto y_str = std::string{};
        auto skip = bool{false};
        std::for_each(std::next(std::begin(text), 1), std::end(text),
                          [&](const auto in) -> void {
                              if (skip) {
                                  return;
                              }

                              if (in >= '0' && in <= '9') {
                                  y_str += in;
                              } else {
                                  y_str = std::string{};
                                  skip = true;
                              }
                          });

        if (!y_str.empty()) {
            y = std::stoi(y_str) - 1;
        }

        if (x == -1 || y == -1) {
            return Board::NO_VERTEX;
        }
        return board.get_vertex(x, y);
    };

    auto parser = Utils::CommandParser(input);

    if (parser.get_count() == 2) {

        const auto color_str = parser.get_command(0)->str;
        const auto vtx_str = parser.get_command(1)->str;

        if (color_str == "B" || color_str == "b" || color_str == "black") {
            color = Board::BLACK;
        } else if (color_str == "W" || color_str == "w" || color_str == "white") {
            color = Board::WHITE;
        }
        vertex = text2vertex(vtx_str);

    } else if (parser.get_count() == 1) {

        const auto vtx_str = parser.get_command(0)->str;
        color = board.get_to_move();
        vertex = text2vertex(vtx_str);
    }

    if (color == Board::INVAL || vertex == Board::NO_VERTEX) {
        return false;
    }

    return play_move(vertex, color);
}

bool GameState::play_move(const int vtx) {
    int to_move = board.get_to_move();
    return play_move(vtx, to_move);
}

void GameState::set_to_move(int color) { 
    board.set_to_move(color);
}

void GameState::display(const size_t strip) const {
    auto res = display_to_string(strip);
    Utils::auto_printf("%s\n", res.c_str());
}

std::string GameState::display_to_string(const size_t strip) const {

    auto out = std::ostringstream{};
    out << std::endl;
    board.board_stream(out, board.get_last_move());
    board.info_stream(out);
    board.prisoners_stream(out);
    board.hash_stream(out);
    m_time_control.time_stream(out);

    for (auto s = size_t{0}; s < strip; ++s) {
        out << std::endl;
    }

    return out.str();
}

std::string GameState::get_sgf_string() const {

    std::ostringstream out;
    const int movenum = board.get_movenum();
    for (int p = 1; p <= movenum; ++p) {
        out << ";";
        auto past_board = m_game_history[p];
        past_board->sgf_stream(out);
    }
    return out.str();
}

std::string GameState::vertex_to_string(int vertex) const {
    return board.vertex_to_string(vertex);
}

void GameState::set_rule(Board::rule_t rule) {
    m_rule = rule;
}

float GameState::final_score(Board::rule_t rule, float addition_komi) const {

    float score = board.area_score(board.get_komi(), rule) - addition_komi;
    float error = 1e-2;
    if (score < error && score > (-error)) {
        score = 0.0f;
    }

    return score;
}

float GameState::final_score(float addition_komi) const {
    return final_score(m_rule, addition_komi);
}

Board::rule_t GameState::get_rule() const {
    return m_rule;
}

int GameState::get_resigned() const {
    return m_resigned;
}

int GameState::get_winner() {

    if (get_resigned() != Board::INVAL) {
        if (get_resigned() == Board::EMPTY) {
            return Board::EMPTY;
        } else {
            return !get_resigned();
        }
    }

    if (board.get_passes() >= 2) {
        float score = final_score();
        if (score > 0.0f) {
            return Board::BLACK;
        } else if (score < 0.0f){
            return Board::WHITE;
        } else {
            return Board::EMPTY;
        }
    }

    return Board::INVAL;
}

bool GameState::isGameOver() const {
    if (get_resigned() != Board::INVAL) {
        return true;
    }
    if (board.get_passes() >= 2) {
        return true;
    }
    return false;
}

void GameState::result_stream(std::ostream &out) {
    if (isGameOver()) {
        if (get_resigned() == Board::EMPTY) {
            out << "Draw";
        } else if (get_resigned() == Board::BLACK) {
            out << "White wins by resigned";
        } else if (get_resigned() == Board::WHITE) {
            out << "Black wins by resigned";
        } else {
            float score = final_score();
            if (score > 0.0f) {
                out << "Black wins +" << score;
            } else if (score < 0.0f){
                score = (-score);
                out << "White wins +" << score;
            } else {
                out << "Draw";
            }
        }
    } else {
        out << "continue...";
    }
}


std::string GameState::result_to_string() {
    auto out = std::ostringstream{};
    result_stream(out);
    return out.str();
}

const std::shared_ptr<Board> GameState::get_past_board(int moves_ago) const {
    const auto movenum = board.get_movenum();
    assert(moves_ago >= 0 && moves_ago <= movenum);
    assert((unsigned)movenum + 1 <= m_game_history.size());
    return m_game_history[movenum - moves_ago];
}

bool GameState::superko() const {

    auto first = std::crbegin(m_kohash_history);
    auto last = std::crend(m_kohash_history);
    auto res = std::find(++first, last, board.get_ko_hash());

    return res != last;
}

void GameState::reset_time() {
    m_time_control.gether_time_settings();
}

void GameState::time_clock() {
    m_time_control.clock();
}

float GameState::get_thinking_time() const {

  return m_time_control.get_thinking_time(
             board.get_to_move(), board.get_boardsize(), board.get_movenum());

}

void GameState::recount_time(const int color) {
    m_time_control.spend_time(color);
}

void GameState::set_time_left(int color, int main_time, int byo_time, int stones) {
    m_time_control.set_time_left(color, main_time, byo_time, stones);
}

bool GameState::is_legal(const int vtx,
                         const int color,
                         Board::avoid_t avoid) const {

    return board.is_legal(vtx, color, avoid);
}

int GameState::get_movenum() const {
    return board.get_movenum();
}

int GameState::get_passes() const {
    return board.get_passes();
}

float GameState::get_komi() const {
    return board.get_komi();
}

void GameState::set_komi(const float komi) {
    board.set_komi(komi);
}

int GameState::get_last_move() const {
    return board.get_last_move();
}
