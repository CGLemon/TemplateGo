#include "SelfPlay.h"
#include <sstream>

SelfPlay::SelfPlay() {
    init();
    loop();
}

void SelfPlay::init() {
    if (m_selfplay_engine == nullptr) {
        m_selfplay_engine = new Engine;
    }
    m_selfplay_engine->initialize();
}

void SelfPlay::loop() {
    while (true) {
        auto input = std::string{};
        if (std::getline(std::cin, input)) {

            auto parser = Utils::CommandParser(input);

            if (!parser.valid()) {
                Utils::gtp_fail("no input command");
                continue;
            }

            if (parser.get_count() == 1 && parser.find("quit")) {
                Utils::gtp_output("exit");
                delete m_selfplay_engine;
                break;
            }

            execute(parser);
        }
    }
}

void SelfPlay::execute(Utils::CommandParser &parser) {

    if (const auto res = parser.find("num-selfplay", 0)) {
        if (parser.get_count() == 2) {
            auto games = parser.get_command(1)->get<int>();
            if (games < 1) {
                games = 1;
            }
            m_max_selfplay_games.store(games);
            Utils::gtp_output("");
        } else {
            Utils::gtp_fail("syntax error : num-selfplay <integral>");
        }
    } else if (const auto res = parser.find("sgfname", 0)) {
        if (parser.get_count() == 2) {
            auto fimename = parser.get_command(1)->str;
            sgf_filename = fimename;
            Utils::gtp_output("");
        } else {
            Utils::gtp_fail("syntax error : sgfname <string>");
        }
    } else if (const auto res = parser.find("dataname", 0)) {
        if (parser.get_count() == 2) {
            auto fimename = parser.get_command(1)->str;
            data_filename = fimename;
            Utils::gtp_output("");
        } else {
            Utils::gtp_fail("syntax error : dataname <string>");
        }
    } else if (const auto res = parser.find("dump-info", 0)) {
        auto out = std::ostringstream{};
        out << std::endl;
        out << "Self-play Games : ";
        out << m_max_selfplay_games.load() << std::endl;
        out << "SGF filename : ";
        out << sgf_filename << std::endl;
        out << "Date filename : ";
        out << data_filename;
        Utils::gtp_output("%s", out.str().c_str());
    } else if (const auto res = parser.find("start", 0)) {
        start_selfplay();
        Utils::gtp_output("");
    } else {
        Utils::gtp_fail("unknown command");
    }

}

void SelfPlay::start_selfplay() {
    for (int g = 0; g < m_max_selfplay_games.load(); ++g) {
        m_selfplay_engine->self_play();
        m_selfplay_engine->dump_sgf(sgf_filename);
        m_selfplay_engine->dump_collect(data_filename);
        m_selfplay_engine->clear_board();
        m_selfplay_engine->clear_cache();
    }
}
