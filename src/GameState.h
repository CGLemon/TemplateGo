#ifndef GAMESTATE_H_INCLUDE
#define GAMESTATE_H_INCLUDE

#include <vector>
#include <memory>
#include <cstdint>
#include <string>

#include "Board.h"

class GameState {
public:
	GameState() = default;

	void init_game(int size, float komi);
	bool superko() const;

	bool play_move(const int vtx);
	bool play_move(const int vtx, const int color);
	bool undo_move();

	void display() const;
	std::string display_to_string() const;

	bool play_textmove(std::string input);
	const std::shared_ptr<Board> get_past_board(int moves_ago) const;

	Board board;

	void set_to_move(int color);
	void exchange_to_move();

	float final_score() const;
private:

	std::vector<std::shared_ptr<Board>> game_history;
	std::vector<std::uint64_t> ko_hash_history;
	int m_resigned;
};



#endif
