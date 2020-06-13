#include <iostream>
#include <memory>
#include <string>

#include "Board.h"
#include "Utils.h"
#include "cfg.h"
#include "config.h"
#include "gtp.h"

using namespace std;


/*
/
/ 所有代碼都利用 clang-format 重新整理過
/ 參數 -style=LLVM
/ 
*/
void normal_loop() {

  auto maingame = std::make_shared<GameState>();

  maingame->init_game(cfg_boardsize, cfg_komi);
  while (true) {
    maingame->display();
    auto input = std::string{};
    Utils::auto_printf("TemplateGo : \n");
    if (std::getline(std::cin, input)) {
      gtp::execute(input, *maingame);
    }
  }
}

int main(int argc, char **argv) {

  gtp::gtp_init_all(argc, argv);
  normal_loop();

  return 0;
}
