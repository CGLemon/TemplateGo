#include "SelfPlay.h"
#include "Random.h"

#include <sstream>
#include <random>

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

    const auto state = m_selfplay_engine->get_state();
    const auto default_komi = state.get_komi();
    const auto boardsize = state.get_boardsize();

    for (int g = 0; g < m_max_selfplay_games.load(); ++g) {
        auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
        if (rng.randfix<10>() < 3) { // 30%
            from_scratch();
        }

        komi_randomize(default_komi, boardsize);
        normal_selfplay();
    }
}

void SelfPlay::normal_selfplay() {
    m_selfplay_engine->self_play();
    m_selfplay_engine->dump_sgf(sgf_filename);
    m_selfplay_engine->dump_collect(data_filename);
    m_selfplay_engine->clear_board();
    m_selfplay_engine->clear_cache();
}

void SelfPlay::from_scratch() {
    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    // Moves 1~6 stones
    const int moves = rng.randfix<6>() + 1;
    for (auto m = 0; m < moves; ++m) {
        m_selfplay_engine->random_playmove();
    }
}

void SelfPlay::komi_randomize(const float center_komi, const int boardsize) {

    const int intersections = boardsize * boardsize;
    const float div = boardsize - 0.0f;

    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    std::normal_distribution<float> dis(0.0f, (float)intersections / div);
    const int res = static_cast<int>(dis(rng) + center_komi);

    m_selfplay_engine->reset_komi((float)res);
}
