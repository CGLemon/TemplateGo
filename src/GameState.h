#ifndef GAMESTATE_H_INCLUDE
#define GAMESTATE_H_INCLUDE

#include <cstdint>
#include <memory>
#include <string>
#include <vector>
#include <iostream>

#include "Board.h"

class GameState {
public:
  GameState() = default;

  void init_game(int size, float komi);
  bool superko() const;

  bool play_move(const int vtx);
  bool play_move(const int vtx, const int color);
  bool undo_move();

  void display() const;
  std::string display_to_string() const;
  std::string vertex_to_string(int vertex) const;

  bool play_textmove(std::string input);
  const std::shared_ptr<Board> get_past_board(int moves_ago) const;

  Board board;

  void set_to_move(int color);
  void exchange_to_move();

  float final_score(Board::rule_t rule);
  float final_score();

  std::string get_sgf_string() const;

  void set_rule(Board::rule_t rule);
  Board::rule_t get_rule() const;

  int get_winner();
  int get_resigned() const;

  bool isGameOver() const;

  void result_stream(std::ostream &out);
  std::string result_to_string();
private:
  Board::rule_t m_rule{Board::rule_t::Tromp_Taylor};

  std::vector<std::shared_ptr<Board>> m_game_history;
  std::vector<std::uint64_t> m_kohash_history;
  int m_resigned;
};

#endif
