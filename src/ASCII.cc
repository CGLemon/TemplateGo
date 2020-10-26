#include "ASCII.h"

#include <functional>
#include <string>
#include <sstream>
#include <iostream>

ASCII::ASCII() {
    init();
    loop();
}

void ASCII::init() {
    if (m_ascii_engine == nullptr) {
        m_ascii_engine = new Engine;
    }
    m_ascii_engine->initialize();
}

void ASCII::loop() {
    while (true) {
        m_ascii_engine->display();
        std::cout << PROGRAM << ": ";
        auto input = std::string{};

        if (std::getline(std::cin, input)) {

            auto parser = Utils::CommandParser(input);

            if (!parser.valid()) {
                std::cout << " no input command" << std::endl;
                continue;
            }

            if (parser.get_count() == 1 && parser.find("quit")) {
                std::cout << " exit " << std::endl;
                delete m_ascii_engine;
                break;
            }
            std::cout << execute(parser) << std::endl;
        }
    }
}


std::string ASCII::execute(Utils::CommandParser &parser) {

    auto out = std::ostringstream{};
    const auto lambda_syntax_not_understood =
        [&](Utils::CommandParser &p, size_t ignore) -> void {

        if (p.get_count() <= ignore) { return; }
        out << p.get_commands(ignore)->str << " ";
        out << ": syntax not understood" << std::endl;
    };

    if (const auto res = parser.find("play", 0)) {
        lambda_syntax_not_understood(parser, 3);
        if (const auto in = parser.get_slice(1, 3)) {
            out << m_ascii_engine->play_textmove(in->str);
        } else {
            out << "syntax error : play [B/W] <vertex>";
        }
    } else if (const auto res = parser.find("undo", 0)) {
        lambda_syntax_not_understood(parser, 1);
        out << m_ascii_engine->undo_move();
    } else if (const auto res = parser.find("showboard", 0)) {
        lambda_syntax_not_understood(parser, 1);
        out << m_ascii_engine->showboard();
    } else if (const auto res = parser.find("raw-nn", 0)) {
        lambda_syntax_not_understood(parser, 1);
        out << m_ascii_engine->nn_rawout();
    } else if (const auto res = parser.find("komi", 0)) {
        lambda_syntax_not_understood(parser, 2);
        if (const auto in = parser.get_commands(1)) {
            out << m_ascii_engine->reset_komi(std::stof(in->str));
        } else {
            out << "syntax error : komi <float>";
        }
    } else if (const auto res = parser.find("boardsize", 0)) {
        lambda_syntax_not_understood(parser, 2);
        if (const auto in = parser.get_commands(1)) {
            out << m_ascii_engine->reset_boardsize(std::stoi(in->str));
        } else {
            out << "syntax error : boardsize <integral>";
        }
    } else if (const auto res = parser.find("input-pattens", 0)) {
        lambda_syntax_not_understood(parser, 2);
        if (const auto in = parser.get_commands(1)) {
            out << m_ascii_engine->input_features(std::stoi(in->str));
        } else {
            out << "syntax error : input-pattens <integral>";
        }
    } else if (const auto res = parser.find("genmove", 0)) {
        lambda_syntax_not_understood(parser, 2);
        if (parser.get_count() >= 2) {
            auto color = parser.get_command(1)->str;
            if (color == "b" || color == "B" || color == "black") {
                out << m_ascii_engine->think(Board::BLACK);
            } else if (color == "w" || color == "W" || color == "white") {
                out << m_ascii_engine->think(Board::WHITE);
            } else {
                out << m_ascii_engine->think();
            }
        } else {
            out << m_ascii_engine->think();
        }
    } else if (const auto res = parser.find("self-play", 0)) {
        lambda_syntax_not_understood(parser, 1);
        out << m_ascii_engine->self_play();
    } else if (const auto res = parser.find("dump-collect", 0)) {
        lambda_syntax_not_understood(parser, 2);
        if (parser.get_count() == 1) {
            out << m_ascii_engine->dump_collect();
        } else {
            auto filename = parser.get_commands(1)->str;
            out << m_ascii_engine->dump_collect(filename);
        }
    } else if (const auto res = parser.find("dump-sgf", 0)) {
        lambda_syntax_not_understood(parser, 2);
        if (parser.get_count() == 1) {
            out << m_ascii_engine->dump_sgf();
        } else {
            auto filename = parser.get_commands(1)->str;
            out << m_ascii_engine->dump_sgf(filename);
        }
    } else if (const auto res = parser.find("dump-misc-features", 0)) {
        lambda_syntax_not_understood(parser, 1);
        out << m_ascii_engine->misc_features();
    } else if (const auto res = parser.find("nn-benchmark", 0)) {
        lambda_syntax_not_understood(parser, 2);
        if (const auto in = parser.get_commands(1)) {
            out << m_ascii_engine->nn_batchmark(std::stoi(in->str));
        } else {
            out << "syntax error : nn-benchmark <integral>";
        }
    } else {
        out << "unknown command";
    }

    return out.str();
}

