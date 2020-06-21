#ifndef TIMECONTROL_H_INCLUDED
#define TIMECONTROL_H_INCLUDED

#include <array>
#include <chrono>

#include "Utils.h"
#include "Board.h"
#include "string"

using namespace Utils;

class TimeControl {
public:
  TimeControl(float main_time = 60 * 60 * 100, float byo_yomi_time = 0,
              int byo_yomi_stones = 0);

  void reset();
  void check_in_byo();

  void print_time(int color);
  void clock();
  void spend_time(int color);
  
  bool is_overtime(int color) const;
  float get_thinking_time(int color, int boardsize, int num_move) const;  


private:
  float m_maintime;
  float m_byotime;
  int m_byostones;


  std::array<float, 2> m_maintime_left;
  std::array<float, 2> m_byotime_left;
  std::array<int, 2> m_stones_left;
  std::array<bool, 2> m_inbyo;

  Timer m_timer;


  bool one_stone_case(int color) const;
  float one_stone_think_time(int color) const;
  bool main_time_case(int color) const;
  float main_time_think_time(int color, int boardsize, int num_move, float lagbuffer_cs) const;
};

#endif
