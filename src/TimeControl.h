#ifndef TIMECONTROL_H_INCLUDED
#define TIMECONTROL_H_INCLUDED

#include <array>
#include <chrono>
#include <iostream>

#include "Utils.h"
#include "Board.h"
#include "string"

using namespace Utils;

class TimeControl {
public:
    TimeControl() = default;

    void gether_time_settings();
    void gether_time_settings(const int main_time,
                              const int byo_yomi_time,
                              const int byo_yomi_stones);
    void reset();

    void clock();
    void spend_time(const int color);
  
    bool is_overtime(const int color) const;
    float get_thinking_time(int color,
                            int boardsize,
                            int num_move) const;  

    void time_stream(std::ostream &out) const;
    void time_stream(std::ostream &out, int color) const;
    void set_time_left(const int color, const int main_time,
                       const int byo_time, const int stones);

private:
    int m_maintime;
    int m_byotime;
    int m_byostones;

    std::array<float, 2> m_maintime_left;
    std::array<float, 2> m_byotime_left;
    std::array<int, 2> m_stones_left;
    std::array<bool, 2> m_inbyo;

    Timer m_timer;

    void check_in_byo();
    bool one_stone_case(int color) const;
    float one_stone_think_time(int color) const;
    bool main_time_case(int color) const;
    float main_time_think_time(int color, int boardsize, int num_move) const;
};

#endif
