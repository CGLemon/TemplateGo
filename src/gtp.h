#ifndef GTP_H_INCLUDE
#define GTP_H_INCLUDE

#include <string>

#include "GameState.h"
#include "Zobrist.h"
#include "cfg.h"

namespace gtp {

static constexpr int GTP_VERSION = 2;


void gtp_init_all(int argc, char** argv);

static const std::vector<std::string> gtp_commands = {
    "protocol_version",
    "name",
    "version",
    "quit",
    "known_command",
    "list_commands",
    "boardsize",
    "clear_board",
    "komi",
    "play",
    "genmove",
    "showboard",
    "undo"
    ""
};

void execute(std::string input, GameState & state);
bool gtp_execute(std::string input, GameState & state);
bool normal_execute(std::string input, GameState & state);
void gtp_mode();



}


#endif
