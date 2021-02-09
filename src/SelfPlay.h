#ifndef SELFPLAY_H_INCLUDE
#define SELFPLAY_H_INCLUDE

#include "Engine.h"
#include "Utils.h"

#include <memory>
#include <string>
#include <atomic>

class SelfPlay {
public:
    SelfPlay();
    SelfPlay(const SelfPlay&) = delete;
    SelfPlay& operator=(const SelfPlay&) = delete;

private:
    void init();
    void loop();
    void execute(Utils::CommandParser &parser);

    void start_selfplay();
    void normal_selfplay();
    void from_scratch();
    void komi_randomize(const float center_komi, const int boardsize);

    Engine *m_selfplay_engine{nullptr};

    std::atomic<int> m_max_selfplay_games{1};
    std::string sgf_filename{"out.sgf"};
    std::string data_filename{"out.txt"};

};

#endif
