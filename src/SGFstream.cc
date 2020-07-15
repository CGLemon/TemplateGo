#include "SGFstream.h"
#include "Utils.h"
#include "config.h"

#include <sstream>
#include <fstream>

void SGFstream::save_sgf(std::string filename, GameState &state) {
  auto winner = state.get_winner();
  float score = -1;
  if (state.board.get_passes() >= 2) {
    score = state.final_score();
    if (winner == Board::WHITE) {
      score = (-score);
    }
    assert(score >= 0);
  }

  std::ostringstream out;
  sgf_stream(out, state, winner, score);
  
  std::ofstream sgf;

  sgf.open(filename);
  if (sgf.is_open()) {
    sgf << out.str();
  } else {
    Utils::static_printf("Error opening file\n");
  }
  sgf.close();
}



void SGFstream::sgf_stream(std::ostream &out, 
                           GameState &state, int winner, float score) {

  auto ruleToString = [=](Board::rule_t rule){
    if (rule == Board::rule_t::Tromp_Taylor) {
      return "Chinese";
    } else if (rule == Board::rule_t::Jappanese) {
      return "Japanese";
    } else {
      return "Error";
    }
  };


  out << "(";
  out << ";";

  out << "GM[1]";
  out << "FF[4]";

  if (winner == Board::EMPTY) {
    out << "RE[0]";
    assert(score == 0);
  }
  else if (winner != Board::INVAL) {
    auto win = std::string{};
    auto res  = std::string{};
    if (winner == Board::BLACK) {
      win = "B+";
    } else if (winner == Board::WHITE) {
      win = "W+";
    } else {
      win = "Error";
    }
    if (score < 0.f) {
      res = "Resign";
    } else {
      res = std::to_string(score);
    }

    out << "RE[";
    out << win << res;
    out << "]";
  }

  out << "AP[";
  out << PROGRAM_NAME;
  out <<  "]";

  out << "RU[";
  out << ruleToString(state.get_rule());
  out << "]"; 
 
  out << "KM[";
  out << state.board.get_komi();
  out << "]";

  out << "SZ[";
  out << state.board.get_boardsize();
  out << "]";

  out << state.get_sgf_string();
  out << ")";

}

