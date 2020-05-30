#include <memory>
#include <queue>
#include <array>
#include <vector>

#include "Board.h"
#include "config.h"
#include "Utils.h"
#include "Zobrist.h"
#include "cfg.h"

using namespace Utils;

constexpr int Board::RESIGN;
constexpr int Board::PASS;
constexpr int Board::NO_VERTEX;
constexpr int Board::NUM_SYMMETRIES;
constexpr int Board::IDENTITY_SYMMETRY;

std::array<std::array<int, NUM_INTERSECTIONS>, Board::NUM_SYMMETRIES> Board::symmetry_nn_idx_table;
std::array<std::array<int, NUM_VERTICES>, Board::NUM_SYMMETRIES> Board::symmetry_nn_vtx_table;
std::array<int, 8> Board::m_dirs;

void Board::BitBoard::init_bitboard(const int numvertices) {
	int count = numvertices/64;
	int res   = numvertices%64;

	if (res != 0) {
		count++;
	}

	bitboard.reserve(count);
	for (int i = 0; i < count; ++i) {
		bitboard.emplace_back(0);
	}
}


void Board::BitBoard::add_bit(const int vertex) {
	int count = vertex/64;
	int idx   = vertex%64;
	bitboard[count] |= make_bit(idx);
}

void Board::BitBoard::remove_bit(const int vertex) {
	int count = vertex/64;
	int idx   = vertex%64;
	bitboard[count] ^= make_bit(idx);
}

std::uint64_t Board::BitBoard::make_bit(const int idx) {
	assert(idx >= 0 || idx < 64);
	return (0x1 << idx);
}



void Board::Chain::reset_chain() {
	for (int vtx = 0; vtx < NUM_VERTICES+1; vtx++) {
		parent[vtx] = NUM_VERTICES;
		next[vtx]   = NUM_VERTICES;
		stones[vtx] = 0;
		libs[vtx]   = 0;
	}
	libs[NUM_VERTICES] = 16384;
}

void Board::Chain::add_stone(const int vtx, const int lib) {
	next[vtx]     = vtx;
	parent[vtx]   = vtx;
	libs[vtx]     = lib;
	stones[vtx]   = 1;
}

void Board::Chain::display_chain() {
	
	auto_printf("vertex \n");
	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			const int vtx = x+1 + (BOARD_SIZE+2)* (y+1);
			auto_printf("%5d ",vtx);
		}
		auto_printf("\n");
	}
	auto_printf("final: %5d \n",NUM_VERTICES);
	auto_printf("\n");
	auto_printf("next \n");
	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			const int vtx = x+1 + (BOARD_SIZE+2)* (y+1);
			auto_printf("%5d ",next[vtx]);
		}
		auto_printf("\n");
	}
	auto_printf("final: %5d \n",next[NUM_VERTICES]);
	auto_printf("\n");
	auto_printf("parent \n");
	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			const int vtx = x+1 + (BOARD_SIZE+2)* (y+1);
			auto_printf("%5d ",parent[vtx]);
		}
		auto_printf("\n");
	}
	auto_printf("final: %5d \n",parent[NUM_VERTICES]);
	auto_printf("\n");
	auto_printf("libs \n");
	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			const int vtx = x+1 + (BOARD_SIZE+2)* (y+1);
			auto_printf("%5d ",libs[vtx]);
		}
		auto_printf("\n");
	}
	auto_printf("final: %5d \n",libs[NUM_VERTICES]);
	auto_printf("\n");
	auto_printf("stones \n");
	for (int y = 0; y < BOARD_SIZE; y++) {
		for (int x = 0; x < BOARD_SIZE; x++) {
			const int vtx = x+1 + (BOARD_SIZE+2)* (y+1);
			auto_printf("%5d ",stones[vtx]);
		}
		auto_printf("\n");
	}
	auto_printf("final: %5d \n",stones[NUM_VERTICES]);
	auto_printf("\n");
}

void Board::display_chain() {
	m_string.display_chain();
}



std::pair<int, int> Board::get_symmetry(const int x, const int y,
										const int symmetry, const int boardsize) {
  
    assert(x >= 0 && x < boardsize);
    assert(y >= 0 && y < boardsize);
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

	int idx_x = x;
	int idx_y = y;

    if ((symmetry & 4) != 0) {
        std::swap(idx_x, idx_y);
    }

    if ((symmetry & 2) != 0) {
        idx_x = boardsize - idx_x - 1;
    }

    if ((symmetry & 1) != 0) {
        idx_y = boardsize - idx_y - 1;
    }

    assert(idx_x >= 0 && idx_x < boardsize);
    assert(idx_y >= 0 && idx_y < boardsize);
    assert(symmetry != IDENTITY_SYMMETRY || (x==idx_x && y==idx_y));

    return {idx_x, idx_y};
}

void Board::init_symmetry_table(const int boardsize) {
	for (int sym = 0; sym < NUM_SYMMETRIES; ++sym) {
		for (int vtx = 0; vtx < m_numvertices; ++vtx) {
			symmetry_nn_vtx_table[sym][vtx] =  0;
		}
	}

	for (int sym = 0; sym < NUM_SYMMETRIES; ++sym) {
		for (int y = 0; y < boardsize; y++) {
			for (int x = 0; x < boardsize; x++) {
				const auto sym_idx = get_symmetry(x, y, sym, boardsize);
				const int vtx      = get_vertex(x, y);
				const int idx      = get_index(x, y);
				symmetry_nn_idx_table[sym][idx] = get_index(sym_idx.first, sym_idx.second);
				symmetry_nn_vtx_table[sym][vtx] = get_vertex(sym_idx.first, sym_idx.second);
			}
		}
	}
}

void Board::init_dirs(const int boardsize) {
	const int x_shift = boardsize+2;
	m_dirs[0] = (-x_shift);
    m_dirs[1] = (-1);
    m_dirs[2] = (+1);
    m_dirs[3] = (+x_shift);
	m_dirs[4] = (-x_shift-1);
    m_dirs[5] = (-x_shift+1);
    m_dirs[6] = (+x_shift-1);
    m_dirs[7] = (+x_shift+1);
}

void Board::init_bitboard(const int numvertices) {
	m_bitstate[BLACK].init_bitboard(numvertices);
	m_bitstate[WHITE].init_bitboard(numvertices);
}

void Board::reset_board(const int boardsize, const float komi) {
	set_boardsize(boardsize);
	set_komi(komi);
	reset_board_data();

	m_komove   = NO_VERTEX;
	m_lastmove = NO_VERTEX;
	m_tomove = BLACK;
	m_prisoners = {0, 0};
	m_passes = 0;
	m_movenum = 0;

	init_symmetry_table(m_boardsize);
	init_dirs(m_boardsize);
	init_bitboard(m_numvertices);

	m_hash = calc_hash(NO_VERTEX);
	m_ko_hash = calc_ko_hash();

	m_string.reset_chain();
}

void Board::set_boardsize(int boardsize) {
	if (boardsize > BOARD_SIZE || boardsize < MARCRO_MIN_BOARDSIZE) {
		boardsize = BOARD_SIZE;
	}
	m_boardsize = boardsize;
	m_letterboxsize = m_boardsize+2;
	m_numvertices = m_letterboxsize* m_letterboxsize;
	m_intersections = m_boardsize* m_boardsize;
}

void Board::set_komi(const float komi) {
    m_komi_integer = static_cast<int>(komi);
	m_komi_float   = komi - static_cast<float>(m_komi_integer);
	if(m_komi_float < 0.01f && m_komi_float > (-0.01f)){
		m_komi_float = 0.0f;
	}
}


void Board::reset_board_data() {
	
	for (int vtx = 0; vtx < m_numvertices; vtx++) {
		m_state[vtx] = INVAL;
		m_neighbours[vtx] = 0;
	}
	m_empty_cnt = 0;	

	for (int y = 0; y < m_boardsize; y++) {
		for (int x = 0; x < m_boardsize; x++) {
			const int vtx = get_vertex(x, y);
			m_state[vtx]           = EMPTY;
            m_empty_idx[vtx]       = m_empty_cnt;
            m_empty[m_empty_cnt++] = vtx;

			if (x == 0 || x == (m_boardsize-1)) {
				m_neighbours[vtx] += ((1 << BLACK_NBR_SHIFT) 
                                   | (1 << WHITE_NBR_SHIFT) 
                                   | (1 << EMPTY_NBR_SHIFT));
			} else {
				m_neighbours[vtx] += (2 << EMPTY_NBR_SHIFT);
			}

			if (y == 0 || y == (m_boardsize-1)) {
				m_neighbours[vtx] += ((1 << BLACK_NBR_SHIFT) 
                                   | (1 << WHITE_NBR_SHIFT) 
                                   | (1 << EMPTY_NBR_SHIFT));
			} else {
				m_neighbours[vtx] += (2 << EMPTY_NBR_SHIFT);
			}
		}
	}
}

void Board::set_passes(int val) {
	if (val > 4) {val = 4;}
	update_zobrist_pass(val, m_passes);
	m_passes = val;
}

void Board::increment_passes() {
	int ori_passes =  m_passes;
	m_passes++;
	if (m_passes > 4) m_passes = 4;
	update_zobrist_pass(m_passes, ori_passes);
}


bool Board::is_star(const int x, const int y) const {
	const int size = m_boardsize;
	const int point = get_index(x, y);
	int stars[3];
    int points[2];
    int hits = 0;

    if (size % 2 == 0 || size < 9) {
        return false;
    }

    stars[0] = size >= 13 ? 3 : 2;
    stars[1] = size / 2;
    stars[2] = size - 1 - stars[0];

    points[0] = point / size;
    points[1] = point % size;

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 3; j++) {
            if (points[i] == stars[j]) {
                hits++;
            }
        }
    }

    return hits >= 2;
}

std::string Board::columns_to_string(const int bsize) const {
	auto res = std::string{};
    for (int i = 0; i < bsize; i++) {
        if (i < 25) {
            res += (('a' + i < 'i') ? 'a' + i : 'a' + i + 1);
        } else {
            res += (('A' + (i - 25) < 'I') ? 'A' + (i - 25) : 'A' + (i - 25) + 1);
        }
		res += " ";
    }
    res += "\n";
	return res;
}


std::string Board::state_to_string(const vertex_t color, bool is_star) const {

	auto res = std::string{};
	color   == BLACK  ? res += "x":
	color   == WHITE  ? res += "o" :
	is_star == true   ? res += "+" :
	color   == EMPTY  ? res += "." :
	color   == INVAL  ? res += "-" : res += "error";
	return res;
}
std::string Board::spcaces_to_string(const int times) const {
	auto res = std::string{};
	for (int i = 0; i < times; ++i) {
		res += " ";
	}
	return res;
}

std::string Board::board_to_string(const int lastmove) const {

	auto res = std::string{};
	m_boardsize > 9 ? (res += spcaces_to_string(3)) : (res += spcaces_to_string(2));
	res += columns_to_string(m_boardsize);
	for (int y = 0; y < m_boardsize; y++) {

		res += std::to_string(y+1);
		if (y < 9 && m_boardsize > 9) {
			res += spcaces_to_string(1);
		}
		if (lastmove == get_vertex(0, y))
            res += "(";
        else
            res += spcaces_to_string(1);

		for (int x = 0; x < m_boardsize; x++) {
			const int vtx = get_vertex(x, y);
			const auto state = get_state(vtx);
			res += state_to_string(static_cast<vertex_t>(state), is_star(x, y));

			if (lastmove == get_vertex(x, y)) res += ")";
            else if (x != m_boardsize-1 && lastmove == get_vertex(x, y)+1) res += "(";
            else res += spcaces_to_string(1);
		}
		res += std::to_string(y+1);
		res += "\n";
	}
	m_boardsize > 9 ? (res += spcaces_to_string(3)) : (res += spcaces_to_string(2));
	res += columns_to_string(m_boardsize);
	return res;
}

std::string Board::prisoners_to_string() const {
	auto res = std::string{};
	res += "BLACK (X) has captured";
	res += std::to_string(m_prisoners[BLACK]);
	res += "stones\n";
	res += "WHITE (O) has captured";
	res += std::to_string(m_prisoners[WHITE]);
	res += "stones\n";
	return res;
}

std::string Board::hash_to_string() const {
	auto res = std::string{};
	res += "HASH : ";
	res += std::to_string(m_hash);
	res += " | ";
	res += "KO_HASH : ";
	res += std::to_string(m_ko_hash);
	res += "\n";
	return res;
}

void Board::text_display() {
	auto res = board_to_string(m_lastmove);
	auto_printf("%s\n", res.c_str());
}	


std::uint64_t Board::calc_hash(int komove, int sym) {
	std::uint64_t res = calc_ko_hash(sym);
	res ^= Zobrist::zobrist_ko[get_transform_vtx(komove, sym)];
	res ^= Zobrist::zobrist_pris[BLACK][m_prisoners[BLACK]];
    res ^= Zobrist::zobrist_pris[WHITE][m_prisoners[WHITE]];
	res ^= Zobrist::zobrist_pass[m_passes];
	if (m_tomove == BLACK) {
		res ^= Zobrist::zobrist_blacktomove;
	}
	return res;
}

std::uint64_t Board::calc_ko_hash(int sym) {
	std::uint64_t res = Zobrist::zobrist_empty;
	for (int vtx = 0; vtx < m_numvertices; vtx++) {
		if (is_in_board(vtx)) {
			res ^= Zobrist::zobrist[m_state[vtx]][get_transform_vtx(vtx, sym)];	
		}
	}
	return res;
}


int Board::count_pliberties(const int vtx) const {
	return (m_neighbours[vtx] >> (EMPTY_NBR_SHIFT)) & NBR_MASK;
}

bool Board::is_simple_eye(const int vtx, const int color) const {
	return m_neighbours[vtx] & s_eyemask[color] ;
}

bool Board::is_real_eye(const int vtx, const int color) const {

    if (!is_simple_eye(vtx, color)) {
        return false;
    }

    int colorcount[4];

    colorcount[BLACK] = 0;
    colorcount[WHITE] = 0;
    colorcount[INVAL] = 0;

	for (int k = 4; k < 8; k++) {
		const int avtx = vtx + m_dirs[k];
		colorcount[m_state[avtx]]++;
	}

    if (colorcount[INVAL] == 0) {
        if (colorcount[!color] > 1) {
            return false;
        }
    } else {
        if (colorcount[!color]) {
            return false;
        }
    }

    return true;
}


bool Board::is_suicide(const int vtx, const int color) const {
	if (count_pliberties(vtx)) {
        return false;
    }

	for (auto k = 0; k < 4; k++) {
        const int avtx = vtx + m_dirs[k];
		const int libs = m_string.libs[m_string.parent[avtx]];
		const int get_color = m_state[avtx];
		if (get_color == color && libs > 1) {
			return false;
			
		} else if (get_color == !color && libs <= 1) {
			return false;
		}
	}

	return true;
}


void Board::add_stone(const int vtx, const int color) {
	assert(color == BLACK || color == WHITE);
	assert(m_state[vtx] == EMPTY);	

	int nbr_pars[4];
	int nbr_par_cnt = 0;
	m_bitstate[color].add_bit(vtx);
	m_state[vtx] = static_cast<vertex_t>(color);
	update_zobrist(vtx, color, EMPTY);
	
	for (int k = 0; k < 4; k++) {
        const int avtx = vtx + m_dirs[k];
		m_neighbours[avtx] += ((1 << (NBR_SHIFT * color)) - (1 << EMPTY_NBR_SHIFT));

		bool found = false;
		const int ip = m_string.parent[avtx];
		for (int i = 0; i < nbr_par_cnt; i++) {
            if (nbr_pars[i] == ip) {
				found = true;
                break;
            }
        }
		if (!found) {
			m_string.libs[ip]--;
        	nbr_pars[nbr_par_cnt++] = ip;
		}
	}
}


void Board::remove_stone(const int vtx, const int color) {
	assert(color == BLACK || color == WHITE);
	int nbr_pars[4];
	int nbr_par_cnt = 0;

	m_bitstate[color].remove_bit(vtx);
	m_state[vtx] = EMPTY;
	update_zobrist(vtx, EMPTY, color);
	
	for (int k = 0; k < 4; k++) {
        const int avtx = vtx + m_dirs[k];
		m_neighbours[avtx] += ((1 << EMPTY_NBR_SHIFT) - (1 << (NBR_SHIFT * color)));

		bool found = false;
		const int ip = m_string.parent[avtx];
		for (int i = 0; i < nbr_par_cnt; i++) {
            if (nbr_pars[i] == ip) {
				found = true;
                break;
            }
        }
		if (!found) {
			m_string.libs[ip]++;
        	nbr_pars[nbr_par_cnt++] = ip;
		}
	}

}


void Board::merge_strings(const int ip, const int aip) {
	assert(ip != NUM_VERTICES && aip != NUM_VERTICES);
	assert(m_string.stones[ip] >= m_string.stones[aip]);
	
	m_string.stones[ip] += m_string.stones[aip];
	int next_pos = aip;

	do {
		for (int k = 0; k < 4; k++) {
			const int apos = next_pos + m_dirs[k];
			if (m_state[apos] == EMPTY) {
				bool found = false;
				for (int kk = 0; kk < 4; kk++) {
					const int aapos = apos + m_dirs[kk];
					if (m_string.parent[aapos] == ip) {
						found = true;
						break;
					}
				}
				if (!found) {
	                m_string.libs[ip]++;
	            } 
			}
		}

		m_string.parent[next_pos] = ip;
		next_pos = m_string.next[next_pos];
	} while(next_pos != aip);

	std::swap(m_string.next[aip], m_string.next[ip]);
}

int  Board::remove_string(const int ip) {
	int pos = ip;
	int removed = 0;
	int color = m_state[ip];
	
	assert(color != EMPTY);

	do {
		remove_stone(pos, color);
		m_string.parent[pos] = NUM_VERTICES;
			
		m_empty_idx[pos]      = m_empty_cnt;
        m_empty[m_empty_cnt]  = pos;
        m_empty_cnt++;
	
		removed++;

		pos = m_string.next[pos];
	} while (pos != ip);

	return removed;
}	

int Board::update_board(const int vtx, const int color) {
	assert(vtx != Board::PASS && vtx != Board::RESIGN);
	add_stone(vtx, color);
	m_string.add_stone(vtx, count_pliberties(vtx));
	bool is_eyeplay = is_simple_eye(vtx, !color);

	int captured_stones = 0;
    int captured_vtx;	

	for (int k = 0; k < 4; k++) {
        const int avtx = vtx + m_dirs[k];
		const int aip = m_string.parent[avtx];

		if (m_state[avtx] == !color) {
			if (m_string.libs[aip] <= 0) {
		        const int this_captured = remove_string(avtx);
		        captured_vtx = avtx;
		        captured_stones += this_captured;
		    } 

		} else if (m_state[avtx] == color) {
			const int ip = m_string.parent[vtx];
			if (ip != aip) {
				if (m_string.stones[ip] >= m_string.stones[aip]) {
                    merge_strings(ip, aip);
                } else {
                    merge_strings(aip, ip);
                }
			}
		}
	} 
	
	if (captured_stones != 0) {
		const int ori_prisoners = m_prisoners[color];
		const int new_prisoners = ori_prisoners + captured_stones;
		m_prisoners[color] = new_prisoners;
		update_zobrist_pris(color, new_prisoners, ori_prisoners);
	}

	int lastvertex = m_empty[--m_empty_cnt];
    m_empty_idx[lastvertex] = m_empty_idx[vtx];
    m_empty[m_empty_idx[vtx]] = lastvertex;

	if (m_string.libs[m_string.parent[vtx]] == 0) {
        assert(captured_stones == 0);
        remove_string(vtx);
    }
	
	if (captured_stones == 1 && is_eyeplay) {
        assert(m_state[captured_vtx] == EMPTY && !is_suicide(captured_vtx, !color));
        return captured_vtx;
    }	

	return NO_VERTEX;
}

void Board::play_move(const int vtx) {
	play_move(vtx, m_tomove);
}
 
void Board::play_move(const int vtx, const int color) {
	assert(vtx != Board::RESIGN);
	const int ori_komove = m_komove;
	if (vtx == PASS) {
		increment_passes();
        m_komove = NO_VERTEX;
    } else {
		if (get_passes() != 0) {set_passes(0);}
        m_komove = update_board(vtx, color);
    }

	if (m_komove != ori_komove) {
		update_zobrist_ko(m_komove, ori_komove);
	}
	m_lastmove = vtx;
	m_movenum++;

}

bool Board::is_legal(const int vtx, const int color,
						std::uint64_t* super_ko_history, Board::avoid_t avoid) const {

	if (vtx == PASS || vtx == RESIGN) {
		return true;
	}

	if (m_state[vtx] != EMPTY) {
		return false;
	}

	if (is_avoid_to_move(avoid, vtx, color)) {
		return false;
	}

	if (!cfg_allowed_suicide && is_suicide(vtx, color))  {
		return false;
	}

	if (vtx == m_komove) {
		return false;
	}

	if (cfg_pre_block_superko && super_ko_history) {
		auto smi_board = std::make_shared<Board>(*this);
		smi_board->play_move(vtx, color);
		const auto kohash = smi_board->get_ko_hash();
		for (int i = 0; i <= m_movenum; i++) {
			if (kohash == *(super_ko_history+i)) {
				return false;
			}
		}
	}
	return true;
}

int Board::calc_reach_color(int color) const {
	assert(color == BLACK || color == WHITE);
    auto reachable = 0;
    auto bd = std::vector<bool>(m_numvertices, false);
    auto open = std::queue<int>();
    for (auto i = 0; i < m_boardsize; i++) {
        for (auto j = 0; j < m_boardsize; j++) {
            auto vertex = get_vertex(i, j);
            if (m_state[vertex] == color) {
                reachable++;
                bd[vertex] = true;
                open.push(vertex);
            }
        }
    }
    while (!open.empty()) {
        /* colored field, spread */
        auto vertex = open.front();
        open.pop();

        for (auto k = 0; k < 4; k++) {
            auto neighbor = vertex + m_dirs[k];
            if (!bd[neighbor] && m_state[neighbor] == EMPTY) {
                reachable++;
                bd[neighbor] = true;
                open.push(neighbor);
            }
        }
    }
    return reachable;
}

float Board::area_score(float komi) const {
    float white = static_cast<float>(calc_reach_color(WHITE));
    float black = static_cast<float>(calc_reach_color(BLACK));
    return black - white - komi;
}

bool Board::is_avoid_to_move(Board::avoid_t avoid, const int vtx, const int color) const {
	if (avoid == avoid_t::NONE) {
		return false;
	} else if (avoid == avoid_t::REAL_EYE) {
		return is_real_eye(vtx, color);
	}

	return false;
}


void Board::set_to_move(int color) {
	assert(color == BLACK || color == WHITE);
	update_zobrist_tomove(color, m_tomove);
	m_tomove = color;
}

void Board::exchange_to_move() {
	m_tomove = !(m_tomove);
	update_zobrist_tomove(BLACK, WHITE);
}


bool Board::is_in_board(const int vtx) {
	return m_state[vtx] != INVAL;
}


int Board::get_boardsize() const {
	return m_boardsize;
}

float Board::get_komi(float beta) const {
	return beta + m_komi_float + static_cast<float>(m_komi_integer);
}

int Board::get_to_move() const {
	return m_tomove;
}

int Board::get_last_move() const {
	return m_lastmove;
}

std::uint64_t Board::get_ko_hash() const {
	return m_ko_hash;
}

std::uint64_t Board::get_hash() const {
	return m_hash;
}

int Board::get_passes() const {
	return m_passes;
}

int Board::get_movenum() const {
	return m_movenum;
}
