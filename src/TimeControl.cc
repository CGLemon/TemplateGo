#include "TimeControl.h"
#include <cassert>

using namespace Utils;


TimeControl::TimeControl(float main_time, float byo_yomi_time,
                         int byo_yomi_stones) {
  m_maintime = main_time;
  m_byotime = byo_yomi_time;
  m_byostones = byo_yomi_stones;

  if (m_byostones <= 0) {
    m_byotime = 0.f;
    m_byostones = 0;
  }
}

void TimeControl::check_in_byo() {
  for (int i = 0; i < 2; ++i) {
    if (m_maintime_left[i] <= 0.0f) {
      m_inbyo[i] = true;
    } else {
      m_inbyo[i] = false;
    }
  }
}


void TimeControl::reset() {
  m_maintime_left = {m_maintime, m_maintime};
  m_byotime_left = {m_byotime, m_byotime};
  m_stones_left = {m_byostones, m_byostones};

  check_in_byo();
}

void TimeControl::print_time(int color) {
  assert(color == Board::BLACK || color == Board::WHITE);

  check_in_byo();
  if (color == Board::BLACK) {
    auto_printf("Black time: ");
  } else {
    auto_printf("White time: ");
  }

  if (!m_inbyo[color]) {
    const int remaining = static_cast<int>(m_maintime_left[color]);
    const int hours = remaining / 3600;
    const int minutes = (remaining % 3600) / 60;
    const int seconds = remaining % 60;
    auto_printf("%02d:%02d:%02d\n", hours, minutes, seconds);
  } else {
    const int remaining = static_cast<int>(m_byotime_left[color]);
    const int stones_left = m_stones_left[color];
    const int hours = remaining / 3600;
    const int minutes = (remaining % 3600) / 60;
    const int seconds = remaining % 60;
    auto_printf("%02d:%02d:%02d", hours, minutes, seconds);
    auto_printf(", %d stones left\n", stones_left);
  }
}

void TimeControl::clock() {
  m_timer.clock();
}

void TimeControl::spend_time(int color) {
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
    } else {
      assert(m_maintime_left[color] == 0);
      assert(is_overtime(color));
    }
  }
}

bool TimeControl::is_overtime(int color) const {
  assert(color == Board::BLACK || color == Board::WHITE);
  if (m_maintime_left[color] >= 0.0f) {
    return false;
  }
  if (m_byotime_left[color] > 0.0f) {
    return false;
  }
  return true;
}



int estimate_moves_expected(int boardsize, int num_move, size_t div_delta) {
  const int board_div = 5 + div_delta;
  const int base_remaining = (boardsize * boardsize) / board_div;
  const int fast_moves = (boardsize * boardsize) / 6;

  if (num_move < fast_moves) {
    return (base_remaining + fast_moves) - num_move;
  } else {
    return base_remaining;
  }
}


float TimeControl::get_thinking_time(int color, int boardsize, int num_move) const {

  float lagbuffer_cs = 100.f;
  float thinking_time = 0.0f;
  if(!is_overtime(color)) {
    float remaning = m_maintime_left[color] + m_byotime_left[color];
    assert(remaning-thinking_time >= 0.0f);
  }

  if (one_stone_case(color)) {
    return one_stone_think_time(color);
  }
  if (main_time_case(color)) {
    return main_time_think_time(color, boardsize, num_move, lagbuffer_cs);
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
    moves_remaining = moves_remaining > 1 ? moves_remaining : 1;

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
  const float buffer_time = remining / 20.f;
  if (buffer_time > 1.f) {
    return remining - 1.0f;
  }
  return remining - buffer_time;
}

bool TimeControl::main_time_case(int color) const {
  if (!m_inbyo[color] && m_stones_left[color] == 0) {
    return true;
  }
  return false;
}

float TimeControl::main_time_think_time(int color, int boardsize, int num_move, float lagbuffer_cs) const {
  float time_remaining = (m_maintime_left[color] - lagbuffer_cs);
  time_remaining = time_remaining > 0.f ? time_remaining : 0.f;

  int moves_remaining = estimate_moves_expected(boardsize, num_move, 0);
  moves_remaining = moves_remaining > 1 ? moves_remaining : 1;

  float time = time_remaining / (float)moves_remaining;

  return time;
}


