#include <iostream>
#include <sstream>
#include <memory>
#include <string>

#include "Board.h"
#include "Utils.h"
#include "cfg.h"
#include "config.h"
#include "gtp.h"

using namespace std;

void normal_loop() {
  auto maingame = std::make_shared<GameState>();
  maingame->init_game(cfg_boardsize, cfg_komi);
  gtp::init_engine(*maingame);

  while (true) {
    maingame->display();
    auto input = std::string{};
    Utils::auto_printf("TemplateGo : \n");
    if (std::getline(std::cin, input)) {
      gtp::execute(input);
    }
  }
}

/*
void selfplay() {
  auto maingame = std::make_shared<GameState>();
  maingame->init_game(cfg_boardsize, cfg_komi);
  gtp::init_engine(*maingame);

  auto cmd = std::stringstream{};

  cmd << "selfplay";
  cmd << " ";
  cmd << cfg_selfplay_config;

  gtp::execute(cmd.str());

  if (cfg_auto_quit == true) {
    auto quit = std::string{"quit"};
    gtp::execute(quit);
  }
}
*/

int main(int argc, char **argv) {
  gtp::set_up(argc, argv);

  if (gtp::selfplay_command()) {
    //selfplay();
  }

  normal_loop();

  return 0;
}
