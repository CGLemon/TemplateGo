#include "TimeControl.h"

#include <cassert>

TimeControl::TimeControl(float main_time, float byo_yomi_time,
                         int byo_yomi_stones) {
  adjust_time(main_time, byo_yomi_time, byo_yomi_stones);
  reset_clock();
}

void TimeControl::adjust_time(float main_time, float byo_yomi_time,
                              int byo_yomi_stones) {
  m_main_time = main_time;
  m_byo_yomi_time = byo_yomi_time;
  m_byo_yomi_stones = byo_yomi_stones;
}

void TimeControl::reset_clock() {

  m_remaining_time = m_main_time;
  m_periods_left = m_byo_yomi_time;
  m_stones_left = m_byo_yomi_stones;

  if (m_byo_yomi_stones == 0 || m_byo_yomi_time == 0) {
    m_periods_left = 0;
    m_stones_left = 0;
  }

  m_total_remaining_time = m_main_time + m_periods_left;
}

void TimeControl::start_clock() {
  m_start_time = std::chrono::steady_clock::now();
}

float TimeControl::during_seconds() {
  auto end_time = std::chrono::steady_clock::now();
  return std::chrono::duration<float>(end_time - m_start_time).count();
}

bool TimeControl::is_time_out() {
  if (m_remaining_time <= 0 && m_periods_left <= 0) {
    return true;
  }
  return false;
}

void TimeControl::using_seconds(float seconds) {
  float consumption = seconds;

  if (m_remaining_time > 0.0f) {
    if (m_remaining_time >= consumption) {
      m_remaining_time -= consumption;
      consumption = 0.0f;
    } else {
      m_remaining_time = 0.0f;
      consumption -= m_remaining_time;
    }
  }

  if (m_remaining_time <= 0.0f) {
    m_periods_left -= consumption;
    if (!is_time_out()) {
      m_stones_left -= 1;
    }
    if (m_stones_left == 0) {
      m_periods_left = m_byo_yomi_time;
    }
  }

  m_total_remaining_time = m_main_time + m_periods_left;
}

float TimeControl::max_time_to_think(int boardsize, int numnove) {

  int pro_move = 0.5 * (boardsize * boardsize - numnove);

  if (m_stones_left == 0) {
    return m_total_remaining_time / (float)pro_move;

  } else {
    return m_total_remaining_time / (float)(m_stones_left + 1);
  }
}
