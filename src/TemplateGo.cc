#include <iostream>
#include <memory>

#include "config.h"
#include "ASCII.h"
#include "GTP.h"

static void ascii_loop() {
    auto ascii = std::make_shared<ASCII>();
}

static void gtp_loop() {
    auto gtp = std::make_shared<GTP>();
}


const static std::string get_License() {

    auto out = std::ostringstream{};

    out << "    ";
    out << PROGRAM << " " << VERSION << " Copyright (C) 2020  Hung-Zhe, Lin"   << std::endl;

    out << "    This program comes with ABSOLUTELY NO WARRANTY."               << std::endl;
    out << "    This is free software, and you are welcome to redistribute it" << std::endl;
    out << "    under certain conditions; see the COPYING file for details."   << std::endl;

    return out.str();
}

int main(int argc, char **argv) {

    init_basic_parameters();
    auto args = std::make_shared<ArgsParser>(argc, argv);

    auto license = get_License();
    Utils::auto_printf("%s\n", license.c_str());
    args->dump();

    if (option<std::string>("mode") == "ascii") {
        ascii_loop();
    } else if (option<std::string>("mode") == "gtp") {
        gtp_loop();
    }


    return 0;
}
