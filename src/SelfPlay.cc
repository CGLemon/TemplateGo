#include "SelfPlay.h"

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
            break;
        }
    }
}

