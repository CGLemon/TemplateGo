#include <iostream>
#include <memory>
#include <string>

#include "Board.h"
#include "gtp.h"
#include "Utils.h"
#include "config.h"
#include "cfg.h"


using namespace std;



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




int main(int argc, char** argv) {

	gtp::gtp_init_all(argc, argv);

	normal_loop();
	
	return 0;
}


