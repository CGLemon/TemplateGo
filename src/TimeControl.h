#ifndef TIMECONTROL_H_INCLUDED
#define TIMECONTROL_H_INCLUDED

#include <array>
#include <chrono>

class TimeControl {
public:
	TimeControl(float main_time = 60 * 60 * 100,
                float byo_yomi_time = 0, int byo_yomi_stones = 0);

	void adjust_time(float main_time, float byo_yomi_time, int byo_yomi_stones);
	void reset_clock();

	void start_clock();
	float during_seconds();
	
	void using_seconds(float seconds);
	bool is_time_out();
	
	float max_time_to_think(int boardsize, int numnove);

private:
	float m_main_time;
    float m_byo_yomi_time;
    int m_byo_yomi_stones;

	std::chrono::steady_clock::time_point m_start_time;
 
	float m_remaining_time;
	float m_periods_left;
	int m_stones_left;

	float m_total_remaining_time;
	
};




#endif
