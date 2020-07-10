#include <sstream>
#include <string>

#include "Board.h"
#include "Evaluation.h"
#include "Search.h"
#include "Utils.h"
#include "gtp.h"

/*
    A list of all valid GTP2 commands is defined here:
    https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
    GTP is meant to be used between programs. It's not a human interface.



    https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
    包含一系列 GTP2 的協議
    GTP 可以讓程式相互溝通，這不是為人設置的界面

*/
using namespace Utils;

Evaluation* evaluation;
std::shared_ptr<Search> search;


std::string gtp_vertex_parser(int vertex, GameState & state) {
  assert(cfg_boardsize == state.board.get_boardsize());
  const int max_vertex = (cfg_boardsize+2)*(cfg_boardsize+2);
  if ((vertex >= 0 && vertex < max_vertex) || vertex == Board::PASS || vertex == Board::RESIGN) {
    return state.vertex_to_string(vertex);
  } else if (vertex == Board::NO_VERTEX) {
    return std::string{"no-vertex"};
  } 
  return std::string{"vertex-erro"};
}

void gtp::gtp_init_all(int argc, char **argv) {
  Zobrist::init_zobrist();
  init_cfg();

  arg_parser(argc, argv);
  
  auto_printf("Warning! The program version is %s. Don't work well now.\n", PROGRAM_VERSION.c_str());
  auto_printf("All parameters are initialize. %s is ready...\n",
               PROGRAM_NAME.c_str());
}

void gtp::init_network(GameState &state) {
  evaluation = new Evaluation;
  evaluation->initialize_network(cfg_playouts, cfg_weightsfile);
  search = std::make_shared<Search>(state, *evaluation);
}

void gtp::execute(std::string input, GameState & state) {

  bool extend_success = extend_execute(input, state);
  if (extend_success) {
    gtp_printf("\n");
    return;
  }

  bool gtp_success = gtp_execute(input, state);

  if (!gtp_success) {
    gtp_fail_printf("unknown command\n");
    return;
  }
}

bool gtp::extend_execute(std::string input, GameState &state) {

  std::stringstream cmd_stream(input);
  auto cmd = std::string{};
  cmd_stream >> cmd;

  if (cmd == "play") {

    std::string move_string, color, vertex;
    int to_move = state.board.get_to_move();
    if (to_move == Board::BLACK) {
      color = "b";
    } else if (to_move == Board::WHITE) {
      color = "w";
    } else {
      return false;
    }

    cmd_stream >> vertex;

    move_string = color + " " + vertex;

    if (!cmd_stream.fail()) {
      bool success = state.play_textmove(move_string);
      if (success) {
        state.exchange_to_move();
        return true;
      }
    }
  } else if (cmd == "benchmark") {
    std::string playouts_str;
    cmd_stream >> playouts_str;

    if (cmd_stream.fail()) {
      search->benchmark(cfg_playouts);
    } else {
      int playouts;
      if (is_allnumber(playouts_str)) {
        playouts = std::stoi(playouts_str);
      } else {
        playouts = cfg_playouts;
      }
      search->benchmark(playouts);
    }
    return true;
  }

  return false;
}

bool gtp::gtp_execute(std::string input, GameState &state) {

  std::stringstream cmd_stream(input);
  auto cmd = std::string{};

  cmd_stream >> cmd;
  if (cmd == "quit") {
    gtp_printf("\n");
    delete evaluation;
    exit(EXIT_SUCCESS);

  } else if (cmd == "protocol_version") {
    gtp_printf("%d\n", GTP_VERSION);

  } else if (cmd == "showboard") {
    gtp_printf("%s\n", state.display_to_string().c_str());

  } else if (cmd == "name") {
    gtp_printf("%s\n", PROGRAM_NAME.c_str());

  } else if (cmd == "version") {
    gtp_printf("%s\n", PROGRAM_VERSION.c_str());

  } else if (cmd == "known_command") {
    cmd_stream >> cmd;
    for (int i = 0; gtp_commands.size() > i; ++i) {
      if (cmd == gtp_commands[i]) {
        gtp_printf("true\n");
        return true;
      }
    }
    gtp_printf("false\n");

  } else if (cmd == "list_commands") {
    auto outtmp = std::string{};
    for (int i = 0; gtp_commands.size() > i; ++i) {
      outtmp += gtp_commands[i];
      outtmp += "\n";
    }
    gtp_printf("%s", outtmp.c_str());
  } else if (cmd == "clear_board") {
    state.init_game(cfg_boardsize, cfg_komi);
    gtp_printf("\n");

  } else if (cmd == "final_score") {
    auto rule = std::string{};
    cmd_stream >> rule;
    float ftmp;
    if (rule == "jappenese") {
      ftmp = state.final_score(Board::rule_t::Jappanese);
    } else if (rule == "Tromp-Taylor") {
      ftmp = state.final_score(Board::rule_t::Tromp_Taylor);
    } else {
      ftmp = state.final_score();
    }
    if (ftmp < -0.1f) {
      gtp_printf("W+%3.1f", float(fabs(ftmp)));
    } else if (ftmp > 0.1f) {
      gtp_printf("B+%3.1f", ftmp);
    } else {
      gtp_printf("0");
    }

  } else if (cmd == "boardsize") {
    int size;
    cmd_stream >> size;
    if (!cmd_stream.fail()) {
      float old_komi = state.board.get_komi();
      cfg_boardsize = size;
      state.init_game(size, old_komi);
      gtp_printf("\n");
    } else {
      gtp_fail_printf("syntax not understood\n");
    }

  } else if (cmd == "komi") {
    float komi;
    float old_komi = state.board.get_komi();
    cmd_stream >> komi;
    if (!cmd_stream.fail()) {
      if (komi != old_komi) {
        cfg_komi = komi;
        state.board.set_komi(komi);
      }
      gtp_printf("\n");
    } else {
      gtp_fail_printf("syntax not understood\n");
    }

  } else if (cmd == "play") {
    std::string move_string, color, vertex;
    printf("here\n");
    cmd_stream >> color >> vertex;
    move_string = color + " " + vertex;

    if (!cmd_stream.fail()) {
      if (!state.play_textmove(move_string)) {
        gtp_fail_printf("illegal move");
      } else {
        gtp_printf("\n");
      }
    } else {
      gtp_fail_printf("syntax not understood\n");
    }
  } else if (cmd == "undo") {
    if (state.undo_move()) {
      gtp_printf("\n");
    } else {
      gtp_fail_printf("cannot undo");
    }
  } else if (cmd == "genmove") {
    std::string color;

    cmd_stream >> color;
    int to_move;
    if (cmd_stream.fail()) {
      int move = search->think(Search::strategy_t::NN_UCT);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s\n", res.c_str());
      state.play_move(move);
      state.exchange_to_move();
    } else {
      if (color == "B" || color == "b" || color == "black") {
        to_move = Board::BLACK;
      } else if (color == "W" || color == "w" || color == "white") {
        to_move = Board::WHITE;
      } else {
        gtp_fail_printf("syntax not understood\n");
        return true;
      }
      int move = search->think(Search::strategy_t::NN_UCT);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s\n", res.c_str());
      state.play_move(move, to_move);
    }

  } else if (cmd == "fixed_handicap") {
    gtp_printf("\n");

  } else if (cmd == "time_settings") {

    std::string main_time, byo_yomi_time, byo_yomi_stones;

    cmd_stream >> main_time >> byo_yomi_time >> byo_yomi_stones;
    if (!cmd_stream.fail()) {

      cfg_maintime = std::stoi(main_time);
      cfg_byotime = std::stoi(byo_yomi_time);
      cfg_byostones = std::stoi(byo_yomi_stones);

      gtp_printf("\n");
    } else {
      gtp_fail_printf("syntax not understood\n");
    }
  } else if (cmd == "evaluation") {
    std::string type;
    cmd_stream >> type;
    if (cmd_stream.fail()) {
      int move = search->think(Search::strategy_t::NN_DIRECT);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s\n", res.c_str());
    } else if (type == "nn-uct") {
      int move = search->think(Search::strategy_t::NN_UCT);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s\n", res.c_str());
    } else if (type == "nn-direct") {
      int move = search->think(Search::strategy_t::NN_DIRECT);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s\n", res.c_str());
    } else {
      gtp_fail_printf("syntax not understood\n");
    }
  } else {
    return false;
  }
  return true;
}

void gtp::gtp_mode() {
  cfg_gtp_mode = true;
  cfg_quiet = true;
}
