#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <sstream>

#include "GameState.h"
#include "Utils.h"

void GameState::init_game(int size, float komi) {
  board.reset_board(size, komi);

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

  if (!board.is_legal(vtx, color, m_kohash_history.data())) {
    return false;
  }
  if (vtx == Board::RESIGN) {
    m_resigned = color;
    return true;
  } else {
    board.play_move(vtx, color);
  }
  const int movenum = board.get_movenum();
  assert(movenum == m_game_history.size());

  m_game_history.emplace_back(std::make_shared<Board>(board));
  m_game_history.resize(movenum + 1);

  m_kohash_history.emplace_back(board.get_ko_hash());
  m_kohash_history.resize(movenum + 1);

  return true;
}

bool GameState::undo_move() {
  const int movenum = board.get_movenum();
  if (movenum > 0) {
    board = *m_game_history[movenum - 1];
    m_game_history.resize(movenum);
    m_kohash_history.resize(movenum);
    return true;
  }
  return false;
}

bool GameState::play_textmove(std::string input) {

  std::stringstream move_stream(input);
  std::string cmd;
  int cmd_count = 0;
  int color;
  int vertex;
  while (move_stream >> cmd) {
    cmd_count++;
    if (cmd_count == 1) {
      if (cmd == "black" || cmd == "b" || cmd == "B") {
        color = Board::BLACK;
      } else if (cmd == "white" || cmd == "w" || cmd == "W") {
        color = Board::WHITE;
      } else {
        return false;
      }
    } else if (cmd_count == 2) {
      if (cmd == "pass") {
        vertex = Board::PASS;
      } else if (cmd == "resign") {
        vertex = Board::RESIGN;
      } else if (cmd[0] >= 'a' && cmd[0] <= 'z') {
        int x = cmd[0] - 97;
        if (cmd[0] >= 'i') {
          x--;
        }
        int cmd_size = cmd.size();
        std::string y_str;
        for (int i = 1; i < cmd_size; ++i) {
          char alpha = cmd[i];
          if (!(alpha >= '0' && alpha <= '9'))
            return false;
          y_str += alpha;
        }
        if (y_str.size() == 0) {
          return false;
        }
        int y = std::stoi(y_str) - 1;
        vertex = board.get_vertex(x, y);

      } else if (cmd[0] >= 'A' && cmd[0] <= 'z') {
        int x = cmd[0] - 65;
        if (cmd[0] >= 'I') {
          x--;
        }
        int cmd_size = cmd.size();
        std::string y_str;
        for (int i = 1; i < cmd_size; ++i) {
          char alpha = cmd[i];
          if (!(alpha >= '0' && alpha <= '9'))
            return false;
          y_str += alpha;
        }
        if (y_str.size() == 0) {
          return false;
        }
        int y = std::stoi(y_str) - 1;
        vertex = board.get_vertex(x, y);

      } else {
        return false;
      }

    } else {
      return false;
    }
  }
  return play_move(vertex, color);
}

bool GameState::play_move(const int vtx) {
  int to_move = board.get_to_move();
  return play_move(vtx, to_move);
}

void GameState::set_to_move(int color) { board.set_to_move(color); }

void GameState::exchange_to_move() { board.exchange_to_move(); }

void GameState::display() const {
  auto res = display_to_string();
  Utils::auto_printf("%s\n", res.c_str());
}

std::string GameState::display_to_string() const {
  auto out = std::ostringstream{};

  out << std::endl;
  board.prisoners_stream(out);

  out << std::endl;
  board.tomove_stream(out);

  out << std::endl;
  board.board_stream(out, board.get_last_move());

  out << std::endl;
  board.hash_stream(out);

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

float GameState::final_score(Board::rule_t rule) {
  return board.area_score(board.get_komi(), rule);
}

float GameState::final_score() {
  return final_score(m_rule);
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
    float error = 1e-4;
    if (score > error) {
      return Board::BLACK;
    } else if (score < (-error)){
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
      float error = 1e-4;
      if (score > error) {
        out << "Black wins +" << score;
      } else if (score < (-error)){
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
  const int movenum = board.get_movenum();
  assert(moves_ago >= 0 && (unsigned)moves_ago <= movenum);
  assert(movenum + 1 <= m_game_history.size());
  return m_game_history[movenum - moves_ago];
}

bool GameState::superko() const {

  auto first = crbegin(m_kohash_history);
  auto last = crend(m_kohash_history);
  auto res = std::find(++first, last, board.get_ko_hash());

  return (res != last);
}
