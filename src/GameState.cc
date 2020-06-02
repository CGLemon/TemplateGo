#include <algorithm>
#include <iterator>
#include <memory>
#include <cassert>
#include <sstream>

#include "GameState.h"
#include "Utils.h"

void GameState::init_game(int size, float komi) {
    board.reset_board(size, komi);

    game_history.clear();
    game_history.emplace_back(std::make_shared<Board>(board));
	
	ko_hash_history.clear();
	ko_hash_history.emplace_back(board.get_ko_hash());

    m_resigned = Board::EMPTY;
}



bool GameState::play_move(const int vtx, const int color) {
	
	if (m_resigned != Board::EMPTY) {
		return false;
	}

	if (!board.is_legal(vtx, color, ko_hash_history.data())) {
		return false;
	}
	if (vtx == Board::RESIGN) {
        m_resigned = color;
		return true;
    } else {
        board.play_move(vtx, color);
    }
	const int movenum = board.get_movenum();
	assert(movenum == game_history.size());
	
    game_history.emplace_back(std::make_shared<Board>(board));
	game_history.resize(movenum+1);	

	ko_hash_history.emplace_back(board.get_ko_hash());
	ko_hash_history.resize(movenum+1);

	return true;
}

bool GameState::undo_move() {
	const int movenum = board.get_movenum();
	if (movenum > 0) {
        board = *game_history[movenum-1];
		game_history.resize(movenum);
		ko_hash_history.resize(movenum);
		return true;
    }
	return false; 
	
}

bool GameState::play_textmove(std::string input) {
	
	std::stringstream move_string(input);
	std::string cmd;
	int cmd_count = 0;
	int color;
	int vertex;
	while (move_string >> cmd) {
		cmd_count++;
		if (cmd_count == 1) {
			if (cmd == "black" || cmd == "b" || cmd == "B") {
				color = Board::BLACK;
			} else if (cmd == "white" || cmd == "w" || cmd == "W") {
				color = Board::WHITE;
			} else {
				return false;
			}
		} else if (cmd_count == 2) {
			if (cmd == "pass") {
				vertex = Board::PASS;
			} else if (cmd == "resign") {
				vertex = Board::RESIGN;
			} else if (cmd[0] >= 'a' && cmd[0] <= 'z') {
				int x = cmd[0] - 97;
				if (cmd[0] >= 'i') {x--;} 
				int cmd_size = cmd.size();
				std::string y_str;
				for (int i = 1; i < cmd_size; ++i) {
					char alpha = cmd[i];
					if (!(alpha >= '0' && alpha <= '9')) return false;
					y_str += alpha;
				}
				if (y_str.size() == 0) {
					return false;
				} 
				int y = std::stoi(y_str)-1;
				vertex = board.get_vertex(x, y);

			} else if (cmd[0] >= 'A' && cmd[0] <= 'z') {
				int x = cmd[0] - 65;
				if (cmd[0] >= 'I') {x--;} 
				int cmd_size = cmd.size();
				std::string y_str;
				for (int i = 1; i < cmd_size; ++i) {
					char alpha = cmd[i];
					if (!(alpha >= '0' && alpha <= '9')) return false;
					y_str += alpha;
				}
				if (y_str.size() == 0) {
					return false;
				} 
				int y = std::stoi(y_str)-1;
				vertex = board.get_vertex(x, y);

			} else {
				return false;
			}
		
		} else {
			return false;
		}
	}
	return play_move(vertex, color);
}

bool GameState::play_move(const int vtx) {
	int to_move = board.get_to_move();
	return play_move(vtx, to_move);
}

void GameState::set_to_move(int color) {
	board.set_to_move(color);
}

void GameState::exchange_to_move() {
	board.exchange_to_move();
}


void GameState::display() const {
	auto res = display_to_string();
	Utils::auto_printf("%s\n",res.c_str());
}

std::string GameState::display_to_string() const {
	auto res = std::string{};
	res += "\n";
	res += board.prisoners_to_string();
	res += "\n";
	res += board.board_to_string(board.get_last_move());
	res += "\n";
	res += board.hash_to_string();
	return res;
}

float GameState::final_score(Board::rule_t rule) {
	return board.area_score(board.get_komi(), rule);
}

const std::shared_ptr<Board> GameState::get_past_board(int moves_ago) const {
	const int movenum = board.get_movenum();
    assert(moves_ago >= 0 && (unsigned)moves_ago <= movenum);
    assert(movenum + 1 <= game_history.size());
    return game_history[movenum - moves_ago];
}

bool GameState::superko() const {
	
	auto first = crbegin(ko_hash_history);
    auto last = crend(ko_hash_history);
    auto res = std::find(++first, last, board.get_ko_hash());

    return (res != last);
}
