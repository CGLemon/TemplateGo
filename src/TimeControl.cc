#include "TimeControl.h"
#include "config.h"

#include <iomanip>
#include <cassert>

using namespace Utils;

void TimeControl::gether_time_settings() {

    gether_time_settings(option<int>("maintime"),
                         option<int>("byotime"),
                         option<int>("byostones"));
}

void TimeControl::gether_time_settings(const int main_time,
                                       const int byo_yomi_time,
                                       const int byo_yomi_stones) {

    m_maintime = main_time;
    m_byotime = byo_yomi_time;
    m_byostones = byo_yomi_stones;

    if (m_byostones <= 0 || m_byotime <= 0) {
        m_byotime = 0;
        m_byostones = 0;
    }

    reset();
}


void TimeControl::check_in_byo() {

    m_inbyo[Board::BLACK] = (m_maintime_left[Board::BLACK] <= 0.0f);
    m_inbyo[Board::WHITE] = (m_maintime_left[Board::WHITE] <= 0.0f);
}


void TimeControl::reset() {

    m_maintime_left = {(float)m_maintime, (float)m_maintime};
    m_byotime_left = {(float)m_byotime, (float)m_byotime};
    m_stones_left = {m_byostones, m_byostones};
    check_in_byo();
}

void TimeControl::time_stream(std::ostream &out) const {
    time_stream(out, Board::BLACK);
    out << " | ";
    time_stream(out, Board::WHITE);
}

void TimeControl::time_stream(std::ostream &out, int color) const {

    assert(color == Board::BLACK || color == Board::WHITE);

    if (color == Board::BLACK) {
        out << "Black time: ";
    } else {
        out << "White time: ";
    }

    if (!m_inbyo[color]) {
       const int remaining = static_cast<int>(m_maintime_left[color]);
       const int hours = remaining / 3600;
       const int minutes = (remaining % 3600) / 60;
       const int seconds = remaining % 60;
       out << std::setw(2) << hours << ":";
       out << std::setw(2) << std::setfill('0') << minutes << ":";
       out << std::setw(2) << std::setfill('0') << seconds;
    } else {
       const int remaining = static_cast<int>(m_byotime_left[color]);
       const int stones_left = m_stones_left[color];
       const int hours = remaining / 3600;
       const int minutes = (remaining % 3600) / 60;
       const int seconds = remaining % 60;

       out << std::setw(2) << hours << ":";
       out << std::setw(2) << std::setfill('0') << minutes << ":";
       out << std::setw(2) << std::setfill('0') << seconds << ", ";;
       out << "Stones left : " << stones_left;
    }
    out << std::setfill(' ');
}


void TimeControl::clock() {
    m_timer.clock();
}

void TimeControl::spend_time(const int color) {

    assert(color == Board::BLACK || color == Board::WHITE);
    float spend = m_timer.get_duration();

    if (!m_inbyo[color]) {
        if (m_maintime_left[color] > spend) {
            m_maintime_left[color] -= spend;
            spend = 0.0f;
        } else if (m_maintime_left[color] == spend) {
            m_maintime_left[color] = 0.0f;
            spend = 0.0f;
            m_inbyo[color] = true;
        } else {
            spend -= m_maintime_left[color];
            m_maintime_left[color] = 0.0f;
            m_inbyo[color] = true;
        }
    }

    if (m_inbyo[color] && spend > 0.0f) {
        m_byotime_left[color] -= spend;
        m_stones_left[color] --;

        if (m_stones_left[color] == 0 && m_byotime_left[color] > 0.0f) {
            m_byotime_left[color] = m_byotime;
            m_stones_left[color] = m_byostones;
        } else {
            assert(m_maintime_left[color] == 0);
            assert(!is_overtime(color));
        }
    }

    check_in_byo();
}

bool TimeControl::is_overtime(const int color) const {

    assert(color == Board::BLACK || color == Board::WHITE);
    if (m_maintime_left[color] > 0.0f) {
        return false;
    }
    if (m_byotime_left[color] > 0.0f) {
        return false;
    }
    return true;
}

int estimate_moves_expected(int boardsize, int num_move, int div_delta) {

    const int board_div = 5 + div_delta;
    const int num_intersections = (boardsize * boardsize);

    const int base_remaining = num_intersections / board_div;
    const int fast_moves = num_intersections / 6;
    const int moves_buffer = (num_intersections / 9);

    int estimate_moves = 0;
    if (num_move < fast_moves) {
        estimate_moves = base_remaining + fast_moves - num_move;
    } else {
        estimate_moves = base_remaining - num_move ;
    }

    if (estimate_moves < moves_buffer) {
        estimate_moves = moves_buffer;
    }

    return estimate_moves;
}

void TimeControl::set_time_left(const int color, const int main_time,
                                const int byo_time, const int stones) {

    assert(color == Board::BLACK || color == Board::WHITE);
    m_maintime_left[color] = static_cast<float>(main_time);
    m_byotime_left[color] = static_cast<float>(byo_time);
    m_stones_left[color] = stones;
    check_in_byo();
}

float TimeControl::get_thinking_time(int color, int boardsize, int num_move) const {

    float lagbuffer_cs = option<float>("lagbuffer");
    float thinking_time = 0.0f;
    if(!is_overtime(color)) {
        float remaning = m_maintime_left[color] + m_byotime_left[color];
        assert(remaning-thinking_time >= 0.0f);
    }

    if (one_stone_case(color)) {
        return one_stone_think_time(color);
    }
    if (main_time_case(color)) {
        return main_time_think_time(color, boardsize, num_move);
    }
    assert(m_byostones != 0);
  
    if (m_inbyo[color]) {
        thinking_time = m_byotime_left[color] / (float)m_stones_left[color];
    } else {
        float byo_extra = m_byotime / (float)m_byostones;
        float time_remaining = m_maintime_left[color] + byo_extra - lagbuffer_cs;
        time_remaining = time_remaining > 0.f ? time_remaining : 0.f;

        int extra_time_per_move = byo_extra - lagbuffer_cs;
        extra_time_per_move = extra_time_per_move > 0.f ? extra_time_per_move : 0.f;

        int moves_remaining = estimate_moves_expected(boardsize, num_move, 0);
        float base_time = time_remaining  / moves_remaining;
        float inc_time = extra_time_per_move;
        thinking_time  = base_time + inc_time;
    }

    return thinking_time;
}

bool TimeControl::one_stone_case(int color) const {

    if (m_inbyo[color] && m_stones_left[color] == 1) {
        return true;
    }
    return false;
}

float TimeControl::one_stone_think_time(int color) const {

    const float remining = m_byotime_left[color];
    const float lagbuffer = remining / 20.f;
    if (lagbuffer > 1.f) {
        return remining - 1.0f;
    }
    return remining - lagbuffer;
}

bool TimeControl::main_time_case(int color) const {

    if (!m_inbyo[color] && m_stones_left[color] == 0) {
        return true;
    }
    return false;
}

float TimeControl::main_time_think_time(int color, int boardsize, int num_move) const {

    const float lagbuffer = 0.1f;
    const float remining = m_maintime_left[color];
    int addition_threads = option<int>("threads") - 1;
    if (addition_threads < 0) {
        addition_threads = 0;
    }

    const float zero_think_lagbuffer = 1.0f + 1.2 * addition_threads * lagbuffer;
    if (remining <= zero_think_lagbuffer) {
        return 0.0f;
    }

    int moves_remaining = estimate_moves_expected(boardsize, num_move, -3);
  
    float time = remining / (float)moves_remaining;

    if (time - lagbuffer < 0) {
        time = 0.f;
    }
    return time;
}
