#include <iostream>
#include <memory>

#include "config.h"
#include "ASCII.h"
#include "GTP.h"
#include "SelfPlay.h"

static void ascii_loop() {
    auto ascii = std::make_shared<ASCII>();
}

static void gtp_loop() {
    auto gtp = std::make_shared<GTP>();
}

static void selfplay_loop() {
    auto gtp = std::make_shared<SelfPlay>();
}

const static std::string get_License() {

    auto out = std::ostringstream{};
    return out.str();
}

int main(int argc, char **argv) {

    init_basic_parameters();
    auto args = std::make_shared<ArgsParser>(argc, argv);

    // auto license = get_License();
    // Utils::auto_printf("%s\n", license.c_str());
    args->dump();

    if (option<std::string>("mode") == "ascii") {
        ascii_loop();
    } else if (option<std::string>("mode") == "gtp") {
        gtp_loop();
    } else if (option<std::string>("mode") == "selfplay") {
        selfplay_loop();
    }


    return 0;
}
