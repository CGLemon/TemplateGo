#include <sstream>
#include <string>
#include <vector>

#include "SGFstream.h"
#include "Board.h"
#include "Search.h"
#include "Utils.h"
#include "gtp.h"
#include "Model.h"
#include "config.h"


/*
    A list of all valid GTP2 commands is defined here:
    https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
    GTP is meant to be used between programs. It's not a human interface.



    https://www.lysator.liu.se/~gunnar/gtp/gtp2-spec-draft2/gtp2-spec.html
    包含一系列 GTP2 的協議
    GTP 可以讓程式相互溝通，這不是為人設置的界面

*/
using namespace Utils;

std::string gtp_vertex_parser(int vertex, GameState &state) {
  assert(cfg_boardsize == state.board.get_boardsize());
  const int max_vertex = (cfg_boardsize+2)*(cfg_boardsize+2);
  if ((vertex >= 0 && vertex < max_vertex) || vertex == Board::PASS || vertex == Board::RESIGN) {
    return state.vertex_to_string(vertex);
  } else if (vertex == Board::NO_VERTEX) {
    return std::string{"no-vertex"};
  } 
  return std::string{"vertex-error"};
}


Engine *gtp::engine = nullptr;


void gtp::set_up(int argc, char **argv) {
  Zobrist::init_zobrist();
  init_cfg();

  arg_parser(argc, argv);
  
  auto_printf("Warning! The program version is %s. Don't work well now.\n", PROGRAM_VERSION.c_str());
}

bool gtp::selfplay_command() {
  if (cfg_selfplay_config != "_NO_CONFIG_FILE_" && cfg_selfplay_agent) {
    return true;
  }

  return false;
}

void gtp::init_engine(GameState &state) {
  if (!engine) {
    engine = new Engine(state);
    engine->init();
  }
  auto_printf("All parameters are initialize. %s is ready...\n",
               PROGRAM_NAME.c_str());
}

void gtp::execute(std::string input) {

  bool gtp_success = gtp_execute(input);

  if (!gtp_success) {
    gtp_fail_printf("unknown command");
    return;
  }
}

bool gtp::gtp_execute(std::string input) {

  GameState &state = engine->m_state;
  std::stringstream cmd_stream(input);
  auto cmd = std::string{};

  cmd_stream >> cmd;
  if (cmd == "quit") {
    gtp_printf("");
    delete engine;
    exit(EXIT_SUCCESS);

  } else if (cmd == "protocol_version") {
    gtp_printf("%d", GTP_VERSION);

  } else if (cmd == "showboard") {
    gtp_printf("%s", state.display_to_string().c_str());

  } else if (cmd == "name") {
    gtp_printf("%s", PROGRAM_NAME.c_str());

  } else if (cmd == "version") {
    gtp_printf("%s", PROGRAM_VERSION.c_str());

  } else if (cmd == "known_command") {
    cmd_stream >> cmd;
    for (auto i = size_t{0}; gtp_commands.size() > i; ++i) {
      if (cmd == gtp_commands[i]) {
        gtp_printf("true\n");
        return true;
      }
    }
    gtp_printf("false");

  } else if (cmd == "list_commands") {
    auto outtmp = std::string{};
    for (auto i = size_t{0}; i < gtp_commands.size(); ++i) {
      outtmp += gtp_commands[i];
      outtmp += "\n";
    }
    gtp_printf("%s", outtmp.c_str());
  } else if (cmd == "clear_board") {
    state.init_game(cfg_boardsize, cfg_komi);
    gtp_printf("");

  } else if (cmd == "clear_cache") {
    engine->evaluation->clear_cache();
    gtp_printf("");
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
    if (ftmp < 0.0f) {
      gtp_printf("W+%3.1f", float(fabs(ftmp)));
    } else if (ftmp > 0.0f) {
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
      gtp_printf("");
    } else {
      gtp_fail_printf("syntax not understood");
    }

  } else if (cmd == "komi") {
    float komi;
    float old_komi = state.board.get_komi();
    assert(old_komi == cfg_komi);
    cmd_stream >> komi;
    if (!cmd_stream.fail()) {
      if (komi != old_komi) {
        cfg_komi = komi;
        state.board.set_komi(komi);
      } 
      gtp_printf("komi %.2f >> %.2f", old_komi, cfg_komi);
    } else {
      gtp_fail_printf("syntax not understood");
    }
  } else if (cmd == "get_komi") {
    float komi = state.board.get_komi();
    assert(komi == cfg_komi);
    gtp_printf("%.1f", komi);

  } else if (cmd == "play") {
    std::string move_string, color, vertex;
    cmd_stream >> color >> vertex;
    move_string = color + " " + vertex;

    if (!cmd_stream.fail()) {
      if (!state.play_textmove(move_string)) {
        gtp_fail_printf("illegal move");
      } else {
        gtp_printf("");
      }
    } else {
      gtp_fail_printf("syntax not understood");
    }
  } else if (cmd == "undo") {
    if (state.undo_move()) {
      gtp_printf("");
    } else {
      gtp_fail_printf("cannot undo");
    }
  } else if (cmd == "genmove") {
    std::string color;

    cmd_stream >> color;
    int to_move;
    if (cmd_stream.fail()) {
      int move = engine->think(Search::strategy_t::NN_UCT);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s", res.c_str());
      state.play_move(move);
    } else {
      if (color == "B" || color == "b" || color == "black") {
        to_move = Board::BLACK;
      } else if (color == "W" || color == "w" || color == "white") {
        to_move = Board::WHITE;
      } else {
        gtp_fail_printf("syntax not understood");
        return true;
      }

      state.set_to_move(to_move);
      auto search_strategy = Search::strategy_t::NN_UCT;
      int move = engine->think(search_strategy);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s", res.c_str());
      state.play_move(move, to_move);
    }

  } else if (cmd == "fixed_handicap") {
    gtp_printf("");

  } else if (cmd == "time_settings") {

    std::string main_time, byo_yomi_time, byo_yomi_stones;

    cmd_stream >> main_time >> byo_yomi_time >> byo_yomi_stones;
    if (!cmd_stream.fail()) {

      cfg_maintime = std::stoi(main_time);
      cfg_byotime = std::stoi(byo_yomi_time);
      cfg_byostones = std::stoi(byo_yomi_stones);
      state.reset_time();
      gtp_printf("");
    } else {
      gtp_fail_printf("syntax not understood");
    }
  } else if (cmd == "time_left") {

    std::string color, main_time, byo_time;
    cmd_stream >> color >> main_time >> byo_time;

    int to_move, m_time, b_time;
    if (!cmd_stream.fail()) {
      bool success = true;
      if (color == "B" || color == "b" || color == "black") {
        to_move = Board::BLACK;
      } else if (color == "W" || color == "w" || color == "white") {
        to_move = Board::WHITE;
      } else {
        success = false;
      } 
      if (success) {
        m_time = std::stoi(main_time);
        b_time = std::stoi(byo_time);
        state.set_time_left(to_move, m_time, b_time);
      } else {
        gtp_fail_printf("syntax not understood");
      }
    } else {
      gtp_fail_printf("syntax not understood");
    }

  } else if (cmd == "evaluation") {
    std::string type;
    cmd_stream >> type;
    if (cmd_stream.fail()) {
      int move = engine->think(Search::strategy_t::NN_DIRECT);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s", res.c_str());
    } else if (type == "nn-uct") {
      int move = engine->think(Search::strategy_t::NN_UCT);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s", res.c_str());
    } else if (type == "nn-direct") {
      int move = engine->think(Search::strategy_t::NN_DIRECT);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s", res.c_str());
    } else {
      gtp_fail_printf("syntax not understood");
    }
  } else if (cmd == "benchmark") {
    std::string playouts_str;
    cmd_stream >> playouts_str;

    if (cmd_stream.fail()) {
      engine->benchmark(cfg_playouts);
    } else {
      int playouts;
      if (is_allnumber(playouts_str)) {
        playouts = std::stoi(playouts_str);
      } else {
        playouts = cfg_playouts;
      }
      engine->benchmark(playouts);
    }
    gtp_printf("");
  } else if (cmd == "printsgf") {
    auto fname = std::string{};
    cmd_stream >> fname;
    if (cmd_stream.fail()) {
      auto res = state.get_sgf_string();
      gtp_printf("%s", res.c_str());
    } else {
      SGFstream::save_sgf(fname, state, true);
      gtp_printf("");
    }
  } else if (cmd == "get_movenum") {
    const int res = state.board.get_movenum();
    gtp_printf("%d", res);

  } else if (cmd == "auto") {
    while(!state.isGameOver()) {
      int move = engine->think(Search::strategy_t::NN_UCT);
      state.play_move(move);

      auto res = gtp_vertex_parser(move, state);
      auto_printf("move = %s\n", res.c_str());
      state.display();
    }
    engine->trainer->gather_winner(state);

    auto res = state.result_to_string();
    gtp_printf("%s", res.c_str());
  } else if (cmd == "get_winner") {
    int winner = state.get_winner();
    if (winner == Board::BLACK) {
      gtp_printf("Black side is winner");
    } else if (winner == Board::WHITE) {
      gtp_printf("White side is winner");
    } else if (winner == Board::EMPTY) {
      gtp_printf("Draw, no winner");
    } else {
      gtp_printf("Game still is keeping");
    }
  } else if (cmd == "adjust-fair-komi"){
    float fair_komi = static_cast<float>(engine->get_fair_komi());
    float old_komi = state.board.get_komi();
    assert(old_komi == cfg_komi);

    if (fair_komi != old_komi) {
      cfg_komi = fair_komi;
      state.board.set_komi(fair_komi);
    } 
    gtp_printf("komi %.1f >> %.1f", old_komi, cfg_komi);

  } else if (cmd == "default-komi"){
    float default_komi = DEFULT_KOMI;
    float old_komi = state.board.get_komi();
    assert(old_komi == cfg_komi);

    if (default_komi != old_komi) {
      cfg_komi = default_komi;
      state.board.set_komi(default_komi);
    } 
    gtp_printf("komi %.1f >> %.1f", old_komi, cfg_komi);

  }  else if (cmd == "dump-reforcement") {
    auto fname = std::string{};
    cmd_stream >> fname;
    engine->trainer->dump_memory();
    if (cmd_stream.fail()) {
      auto out = std::ostringstream{};
      engine->trainer->data_stream(out);
      gtp_printf("\n%s", out.str().c_str());
    } else {
      engine->trainer->save_data(fname, true);
      engine->trainer->clear_game_steps();
      gtp_printf("");
    }
  } else if (cmd=="nn-input-pattens"){
    auto res = Model::features_to_string(state);
    gtp_printf("%s", res.c_str());

  } else if (cmd == "nn-reload") {
    std::string fname;
    cmd_stream >> fname;
    if (cmd_stream.fail()) {
      gtp_fail_printf("syntax not understood");
    } else {
      engine->evaluation->reload_network(fname);
      gtp_printf("");
    }
  } else if (cmd == "random-move") {
    std::string color;
    cmd_stream >> color;
    int to_move;

    auto search_strategy = Search::strategy_t::RANDOM;
    if (cmd_stream.fail()) {
      int move = engine->think(search_strategy);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s", res.c_str());
      state.play_move(move);
    } else {
      if (color == "B" || color == "b" || color == "black") {
        to_move = Board::BLACK;
      } else if (color == "W" || color == "w" || color == "white") {
        to_move = Board::WHITE;
      } else {
        gtp_fail_printf("syntax not understood");
        return true;
      }

      state.set_to_move(to_move);
      int move = engine->think(search_strategy);
      auto res = gtp_vertex_parser(move, state);
      gtp_printf("%s", res.c_str());
      state.play_move(move, to_move);
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
