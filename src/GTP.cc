#include "GTP.h"
#include "config.h"

GTP::GTP() {
    init();
    loop();
}

void GTP::init() {
    if (m_gtp_engine == nullptr) {
        m_gtp_engine = new Engine;
    }
    m_gtp_engine->initialize();
}

void GTP::loop() {
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
                delete m_gtp_engine;
                break;
            }
            execute(parser);
        }
    }
}

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
                                                      "time_left",
                                                      "final_status_list",
                                                     };

void GTP::execute(Utils::CommandParser &parser) {

    static constexpr auto GTP_VERSION = 2;

    if (const auto res = parser.find("protocol_version", 0)) {
        Utils::gtp_output("%d\n", GTP_VERSION);
    } else if (const auto res = parser.find("name", 0)){
        Utils::gtp_output("%s", PROGRAM.c_str());
    } else if (const auto res = parser.find("version", 0)){
        Utils::gtp_output("%s", VERSION.c_str());
    } else if (const auto res = parser.find("list_commands", 0)){
        auto out = std::ostringstream{};
        for (const auto &o : gtp_commands) {
            out << o << "\n";
        }
        Utils::gtp_output("%s", out.str().c_str());
    } else if (const auto res = parser.find("known_command", 0)) {
        auto cmd = parser.get_command(1)->str;
        for (const auto &o : gtp_commands) {
            if (o == cmd) {
                Utils::gtp_output("true");
            }
        }
        Utils::gtp_output("false");
    } else if (const auto res = parser.find("clear_board", 0)) {
        auto gtp_response = m_gtp_engine->clear_board();
        Utils::gtp_output("%s", gtp_response.c_str());
    } else if (const auto res = parser.find("play", 0)) {
        if (const auto in = parser.get_slice(1, 3)) {
            auto gtp_response = m_gtp_engine->play_textmove(in->str);
            Utils::gtp_output("%s", gtp_response.c_str());
        } else {
            Utils::gtp_fail("syntax error : play [B/W] <vertex>");
        }
    } else if (const auto res = parser.find("undo", 0)) {
        auto gtp_response = m_gtp_engine->undo_move();
        Utils::gtp_output("");
    } else if (const auto res = parser.find("showboard", 0)) {
        auto gtp_response = m_gtp_engine->showboard(0);
        Utils::gtp_output("%s", gtp_response.c_str());
    } else if (const auto res = parser.find("genmove", 0)) {
        auto gtp_response = std::string{};
        if (parser.get_count() >= 2) {
            auto color = parser.get_command(1)->str;
            if (color == "b" || color == "B" || color == "black") {
                gtp_response = m_gtp_engine->think(Board::BLACK);
            } else if (color == "w" || color == "W" || color == "white") {
                gtp_response = m_gtp_engine->think(Board::WHITE);
            } else {
                gtp_response = m_gtp_engine->think();
            }
        } else {
            gtp_response = m_gtp_engine->think();
        }
        Utils::gtp_output("%s", gtp_response.c_str());
    } else if (const auto res = parser.find("komi", 0)) {
        if (const auto in = parser.get_command(1)) {
            auto gtp_response = m_gtp_engine->reset_komi(std::stof(in->str));
            Utils::gtp_output("%s", gtp_response.c_str());
        } else {
            Utils::gtp_fail("syntax error : komi <float>");
        }
    } else if (const auto res = parser.find("boardsize", 0)) {
        if (const auto in = parser.get_command(1)) {
            auto gtp_response = m_gtp_engine->reset_boardsize(std::stoi(in->str));
            Utils::gtp_output("%s", gtp_response.c_str());
        } else {
            Utils::gtp_output("syntax error : boardsize <integral>");
        }
    } else if (const auto res = parser.find("time_settings", 0)) {
        if (parser.get_count() < 4) {
            Utils::gtp_fail("syntax error : time_settings main_time byo_yomi_time byo_yomi_stones");
        } else {
            const auto mtime = parser.get_command(1);
            const auto btime = parser.get_command(2);
            const auto stones = parser.get_command(3);

            auto gtp_response = m_gtp_engine-> time_settings(mtime->get<int>(),
                                                             btime->get<int>(),
                                                             stones->get<int>());
            Utils::gtp_output("%s", gtp_response.c_str());
        }
    } else if (const auto res = parser.find("time_left", 0)) {
        if (parser.get_count() < 4) {
            Utils::gtp_fail("syntax error : time_left color time stones");
        } else {
            const auto color = parser.get_command(1);
            const auto time = parser.get_command(2);
            const auto stones = parser.get_command(3);

            if (color->get<std::string>() == "b" || color->get<std::string>() == "B" || color->get<std::string>() == "black" ||
                color->get<std::string>() == "w" || color->get<std::string>() == "W" || color->get<std::string>() == "white") {

                auto gtp_response = m_gtp_engine-> time_left(color->get<std::string>(),
                                                             time->get<int>(),
                                                             stones->get<int>());
                Utils::gtp_output("%s", gtp_response.c_str());
            } else {
                Utils::gtp_fail("syntax error : invalid color");
            }
        }
    } else if (const auto res = parser.find("final_status_list", 0)) {

        auto success = bool{false};
        auto gtp_response = std::string{};
        if (const auto in = parser.get_command(1)) {
            if (in->str == "alive") {
                gtp_response = m_gtp_engine-> final_status_list(true);
                success = true;
            } else if (in->str == "dead") {
                gtp_response = m_gtp_engine-> final_status_list(false);
                success = true;
            }
        }
        if (success) {
            Utils::gtp_output("%s", gtp_response.c_str());
        } else {
            Utils::gtp_fail("syntax error : final_status_list [alive/dead]");
        }
    } else {
        Utils::gtp_fail("unknown command");
    }
}
