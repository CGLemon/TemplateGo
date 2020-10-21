#ifndef SELFPLAY_H_INCLUDE
#define SELFPLAY_H_INCLUDE

#include "Engine.h"
#include "Utils.h"

#include <memory>
#include <string>

class SelfPlay {
public:
    SelfPlay();
    SelfPlay(const SelfPlay&) = delete;
    SelfPlay& operator=(const SelfPlay&) = delete;

private:
    void init();
    void loop();
    void execute(Utils::CommandParser &parser);

    Engine *m_selfplay_engine{nullptr};

};

#endif
