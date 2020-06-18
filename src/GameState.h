#ifndef GAMESTATE_H_INCLUDE
#define GAMESTATE_H_INCLUDE

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

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

  float final_score(Board::rule_t rule = Board::rule_t::Tromp_Taylor);

private:
  std::vector<std::shared_ptr<Board>> game_history;
  std::vector<std::uint64_t> ko_hash_history;
  int m_resigned;
};

#endif
