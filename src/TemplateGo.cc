#include <iostream>
#include <sstream>
#include <memory>
#include <string>

#include "Board.h"
#include "Utils.h"
#include "config.h"
#include "gtp.h"

using namespace std;

void normal_loop() {
  auto maingame = std::make_shared<GameState>();
  maingame->init_game(cfg_boardsize, cfg_komi);
  gtp::init_engine(*maingame);

  while (true) {
    maingame->display(2);
    auto input = std::string{};
    Utils::auto_printf("TemplateGo : \n");
    if (std::getline(std::cin, input)) {
      gtp::execute(input);
    }
  }
}



int main(int argc, char **argv) {
  gtp::set_up(argc, argv);
  normal_loop();

  return 0;
}
