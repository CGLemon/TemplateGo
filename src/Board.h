#ifndef BOARD_H_INCLUDE
#define BOARD_H_INCLUDE

#include <array>
#include <cstdlib>
#include <cassert>
#include <string>
#include <memory>

#include "config.h"
#include "Zobrist.h"

#define BLACK_NUMBER (0)
#define WHITE_NUMBER (1)
#define EMPTY_NUMBER (2)
#define INVAL_NUMBER (3)


#define NBR_SHIFT (4)
#define BLACK_NBR_SHIFT (BLACK_NUMBER* 4)
#define WHITE_NBR_SHIFT (WHITE_NUMBER* 4)
#define EMPTY_NBR_SHIFT (EMPTY_NUMBER* 4)

#define NBR_MASK (0xf)
#define BLACK_EYE_MASK (4 * (1 << BLACK_NBR_SHIFT))
#define WHITE_EYE_MASK (4 * (1 << WHITE_NBR_SHIFT))


static std::array<int ,2> s_eyemask = {BLACK_EYE_MASK, WHITE_EYE_MASK};


class Board {
public:

	enum class avoid_t {
		NONE, REAL_EYE
	};

	enum vertex_t : std::uint8_t {
		BLACK = BLACK_NUMBER, WHITE = WHITE_NUMBER, EMPTY = EMPTY_NUMBER, INVAL = INVAL_NUMBER
	};

	enum territory_t : std::uint8_t {
        B_STONE = 0, W_STONE = 1, EMPTY_I = 2, INVAL_I = 3,
        DAME = 4,
		SEKI = 5  , SEKI_EYE = 6,
        W_TERR = 7, B_TERR = 8
    };


	static constexpr int NO_VERTEX = 0;
	static constexpr int PASS = NUM_VERTICES+1;
	static constexpr int RESIGN = NUM_VERTICES+2;

	static constexpr int NUM_SYMMETRIES = 8;
	static constexpr int IDENTITY_SYMMETRY = 0;

	static std::array<std::array<int, NUM_INTERSECTIONS>, NUM_SYMMETRIES> symmetry_nn_idx_table;
	static std::array<std::array<int, NUM_VERTICES>, NUM_SYMMETRIES> symmetry_nn_vtx_table;
	static std::array<int, 8> m_dirs;

	void reset_board(const int boardsize, const float komi);
	void set_komi(const float komi);
	void set_boardsize(int boardsize);
	void reset_board_data();

	void set_state(const int vtx, const vertex_t);
	void set_to_move(int color);
	void exchange_to_move();


	bool is_star(const int x, const int y) const;
	std::string board_to_string(const int lastmove = NO_VERTEX) const;
	std::string state_to_string(const vertex_t color, bool is_star) const;
	std::string spcaces_to_string(const int times) const;
	std::string columns_to_string(const int bsize) const;
	std::string prisoners_to_string() const;
	std::string hash_to_string() const;
	void text_display();
	void display_chain();

	std::pair<int, int> get_symmetry(const int x, const int y,
										 const int symmetry, const int boardsize);
	int get_to_move() const;
	int get_last_move() const;
	int get_vertex(const int x, const int y) const;
	int get_index(const int x, const int y) const;
	vertex_t get_state(const int vtx) const;
	vertex_t get_state(int x, int y) const;
	int get_boardsize() const;
	int get_transform_idx(const int idx, const int sym = IDENTITY_SYMMETRY) const;
	int get_transform_vtx(const int vtx, const int sym = IDENTITY_SYMMETRY) const;
	int get_passes() const;
	int get_movenum() const;
	float get_komi(float beta = 0.0f) const;

 	std::uint64_t get_ko_hash() const;
	std::uint64_t get_hash() const;

	std::uint64_t calc_hash(int komove, int sym = IDENTITY_SYMMETRY);
	std::uint64_t calc_ko_hash(int sym = IDENTITY_SYMMETRY);


	void set_passes(int val);
	void increment_passes();

	void update_zobrist(const int vtx, const int new_color, const int old_color);
	void update_zobrist_pris(const int color, const int new_pris, const int old_pris);
	void update_zobrist_tomove(const int new_color, const int old_color);
	void update_zobrist_ko(const int new_komove, const int old_komove);
	void update_zobrist_pass(const int new_pass, const int old_pass);

	int count_pliberties(const int vtx) const;
	bool is_real_eye(const int vtx, const int color) const;
	bool is_simple_eye(const int vtx, const int color) const;
	bool is_suicide(const int vtx, const int color) const;
	void add_stone(const int vtx, const int color);
	void remove_stone(const int vtx, const int color);
	void merge_strings(const int ip, const int aip);
	int remove_string(const int ip);
	int update_board(const int vtx, const int color);

	void play_move(const int vtx, const int color);
	void play_move(const int vtx);
	bool is_legal(const int vtx, const int color, 
                    std::uint64_t* super_ko_history = nullptr, avoid_t avoid = avoid_t::NONE) const;
	bool is_avoid_to_move(avoid_t, const int vtx, const int color) const;

	int calc_reach_color(int color, int spread_color = 0, std::shared_ptr<std::vector<bool>> bd = nullptr, bool is_territory = false) const;
	float area_score(float komi) const;
private:
	class BitBoard {
	public:
		BitBoard() = default;
		BitBoard(const int numvertices) {
			init_bitboard(numvertices);
		}
		void init_bitboard(const int numvertices);		
		void add_bit(const int vertex);
		void remove_bit(const int vertex);		

		std::vector<std::uint64_t> bitboard;
	private:
		std::uint64_t make_bit(const int idx);

	};

	struct Chain {
		std::array<std::uint16_t, NUM_VERTICES+1> next;
		std::array<std::uint16_t, NUM_VERTICES+1> parent;
		std::array<std::uint16_t, NUM_VERTICES+1> libs;
		std::array<std::uint16_t, NUM_VERTICES+1> stones;
		void reset_chain();
		void add_stone(const int vtx, const int lib);
		void display_chain();
	};

	std::array<BitBoard     , 2>            m_bitstate;
	std::array<vertex_t     , NUM_VERTICES> m_state;
	std::array<std::uint16_t, NUM_VERTICES> m_neighbours;
	std::array<std::uint16_t, NUM_VERTICES> m_empty;
	std::array<std::uint16_t, NUM_VERTICES> m_empty_idx;

	std::array<territory_t, NUM_VERTICES> m_territory;
	Chain m_string;

	std::array<int, 2> m_prisoners;

	std::uint64_t m_hash;
	std::uint64_t m_ko_hash;

	int m_tomove;	
	int m_letterboxsize;
	int m_numvertices;
	int m_intersections;
	int m_boardsize;
	int m_lastmove;
	int m_komove;
	int m_empty_cnt;
	int m_passes;
	int m_movenum;
	int m_komi_integer;
	float m_komi_float;

	bool is_in_board(const int vtx);
	void init_symmetry_table(const int boardsize);
	void init_dirs(const int boardsize);
	void init_bitboard(const int numvertices);
};

inline int Board::get_vertex(const int x, const int y) const {
	assert(x >= 0 || x < m_boardsize);
	assert(y >= 0 || y < m_boardsize);
	return (y+1)*m_letterboxsize + (x+1);
}

inline int Board::get_index(const int x, const int y) const {
	assert(x >= 0 || x < m_boardsize);
	assert(y >= 0 || y < m_boardsize);
	return y * m_boardsize + x;
}

inline void Board::set_state(const int vtx, const Board::vertex_t color) {
	m_state[vtx] = color;
}

inline Board::vertex_t Board::get_state(int x, int y) const {
	return m_state[get_vertex(x, y)];
}

inline Board::vertex_t Board::get_state(const int vtx) const {
	return m_state[vtx];
}
inline int Board::get_transform_idx(const int idx, const int sym) const {
	return symmetry_nn_idx_table[sym][idx];
}

inline int Board::get_transform_vtx(const int vtx, const int sym) const {
	return symmetry_nn_vtx_table[sym][vtx];
}

inline void Board::update_zobrist(const int vtx, const int new_color, const int old_color) {
	m_hash ^= Zobrist::zobrist[old_color][vtx];
	m_hash ^= Zobrist::zobrist[new_color][vtx];
	m_ko_hash ^= Zobrist::zobrist[old_color][vtx];
	m_ko_hash ^= Zobrist::zobrist[new_color][vtx];
}

inline void Board::update_zobrist_pris(const int color, const int new_pris, const int old_pris) {
	m_hash ^= Zobrist::zobrist_pris[color][old_pris];
	m_hash ^= Zobrist::zobrist_pris[color][new_pris];
}

inline void Board::update_zobrist_tomove(const int new_color, const int old_color) {
	if (old_color != new_color) {
		m_hash ^= Zobrist::zobrist_blacktomove;
	}
}

inline void Board::update_zobrist_ko(const int new_komove, const int old_komove) {
	m_hash ^= Zobrist::zobrist_ko[old_komove];
	m_hash ^= Zobrist::zobrist_ko[new_komove];
}

inline void Board::update_zobrist_pass(const int new_pass, const int old_pass) {
	m_hash ^= Zobrist::zobrist_pass[old_pass];
	m_hash ^= Zobrist::zobrist_pass[new_pass];
}


#endif
