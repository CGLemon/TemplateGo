#ifndef GTP_H_INCLUDE
#define GTP_H_INCLUDE

#include <string>

#include "GameState.h"
#include "Zobrist.h"
#include "config.h"
#include "Engine.h"

namespace gtp {

static constexpr int GTP_VERSION = 2;

extern Engine *engine;

void set_up(int argc, char **argv);

static const std::vector<std::string> gtp_commands = {"protocol_version",
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
                                                      "undo",
                                                      "time_settings",
                                                      "time_left"};

void init_engine(GameState &state);
void execute(std::string input);
bool gtp_execute(std::string input);
void gtp_mode();

bool selfplay_command();
bool train_command();

} // namespace gtp

#endif
