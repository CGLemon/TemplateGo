#ifndef GAMESTATE_H_INCLUDE
#define GAMESTATE_H_INCLUDE

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "TimeControl.h"
#include "Board.h"

class GameState {
public:
    GameState() = default;

    void init_game(int size, float komi);
    bool superko() const;

    bool play_move(const int vtx);
    bool play_move(const int vtx, const int color);
    bool undo_move();

    void display(const size_t strip = 0) const;
    std::string display_to_string(const size_t strip = 0) const;
    std::string vertex_to_string(int vertex) const;

    bool play_textmove(std::string input);
    const std::shared_ptr<Board> get_past_board(int moves_ago) const;

    Board board;

    void set_to_move(int color);

    float final_score(Board::rule_t rule, float addition_komi = 0) const;
    float final_score(float addition_komi = 0) const;

    std::string get_sgf_string() const;

    void set_rule(Board::rule_t rule);
    Board::rule_t get_rule() const;

    int get_winner();
    int get_resigned() const;

    bool isGameOver() const;

    void result_stream(std::ostream &out);
    std::string result_to_string();
  
    void reset_time();
    void time_clock();
    float get_thinking_time() const;
    void recount_time(const int color);
    void set_time_left(int color, int main_time, int byo_time, int stones);


    bool is_legal(const int vtx,
                  const int color,
                  Board::avoid_t avoid = Board::avoid_t::NONE) const;

    int get_x(const int vtx) const;
    int get_y(const int vtx) const;
    std::pair<int, int> get_xy(const int vtx) const;

    int get_boardsize() const;
    int get_to_move() const;
    int get_intersections() const;
    int get_movenum() const;
    int get_vertex(const int x, const int y) const;
    int get_index(const int x, const int y) const;
    int get_passes() const;
    float get_komi() const;
    void set_komi(const float komi);

private:
    TimeControl m_time_control;

    Board::rule_t m_rule{Board::rule_t::Tromp_Taylor};

    std::vector<std::shared_ptr<Board>> m_game_history;
    std::vector<std::uint64_t> m_kohash_history;
    int m_resigned;
};

inline int GameState::get_boardsize() const {
    return board.get_boardsize();
}

inline int GameState::get_intersections() const {
    return board.get_intersections();
}

inline int GameState::get_to_move() const {
    return board.get_to_move();
}

inline int GameState::get_x(const int vtx) const {
    return board.get_x(vtx);
}

inline int GameState::get_y(const int vtx) const {
    return board.get_y(vtx);
}

inline std::pair<int, int> GameState::get_xy(const int vtx) const {
    return board.get_xy(vtx);
}

inline int GameState::get_vertex(const int x, const int y) const {
    return board.get_vertex(x, y);
}

inline int GameState:: get_index(const int x, const int y) const {
    return board.get_index(x, y);
}

#endif
