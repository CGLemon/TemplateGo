#include <array>
#include <memory>
#include <queue>
#include <vector>
#include <iomanip>
#include <algorithm>

#include "Board.h"
#include "Utils.h"
#include "Zobrist.h"
#include "config.h"

using namespace Utils;

constexpr int Board::RESIGN;
constexpr int Board::PASS;
constexpr int Board::NO_VERTEX;
constexpr int Board::NUM_SYMMETRIES;
constexpr int Board::IDENTITY_SYMMETRY;
constexpr int Board::s_eyemask[2];

std::array<std::array<int, NUM_INTERSECTIONS>, Board::NUM_SYMMETRIES>
    Board::symmetry_nn_idx_table;
std::array<std::array<int, NUM_VERTICES>, Board::NUM_SYMMETRIES>
    Board::symmetry_nn_vtx_table;
std::array<int, 8> Board::m_dirs;

void Board::BitBoard::init(const int numvertices) {

    int cnt = numvertices / 64;
    int res = numvertices % 64;

    if (res != 0) {
        cnt++;
    }

    bitboard.reserve(cnt);
    for (int i = 0; i < cnt; ++i) {
        bitboard.emplace_back(0);
    }
}

void Board::BitBoard::add_bit(const int vertex) {
    int cnt = vertex / 64;
    int idx = vertex % 64;
    bitboard[cnt] |= make_bit(idx);
}

void Board::BitBoard::remove_bit(const int vertex) {
    int cnt = vertex / 64;
    int idx = vertex % 64;
    bitboard[cnt] ^= make_bit(idx);
}

std::uint64_t Board::BitBoard::make_bit(const int idx) {
    assert(idx >= 0 && idx < 64);
    return 1ULL << idx;
}

void Board::Chain::reset_chain() {
    for (int vtx = 0; vtx < NUM_VERTICES + 1; ++vtx) {
        parent[vtx] = NUM_VERTICES;
        next[vtx] = NUM_VERTICES;
        stones[vtx] = 0;
        libs[vtx] = 0;
    }
    libs[NUM_VERTICES] = 16384;
}

void Board::Chain::add_stone(const int vtx, const int lib) {
    next[vtx] = vtx;
    parent[vtx] = vtx;
    libs[vtx] = lib;
    stones[vtx] = 1;
}

void Board::Chain::display_chain() {

    auto_printf("vertex \n");
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            const int vtx = x + 1 + (BOARD_SIZE + 2) * (y + 1);
            auto_printf("%5d ", vtx);
        }
        auto_printf("\n");
    }
    auto_printf("final: %5d \n", NUM_VERTICES);
    auto_printf("\n");
    auto_printf("next \n");
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            const int vtx = x + 1 + (BOARD_SIZE + 2) * (y + 1);
            auto_printf("%5d ", next[vtx]);
        }
        auto_printf("\n");
    }
    auto_printf("final: %5d \n", next[NUM_VERTICES]);
    auto_printf("\n");
    auto_printf("parent \n");
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            const int vtx = x + 1 + (BOARD_SIZE + 2) * (y + 1);
            auto_printf("%5d ", parent[vtx]);
        }
        auto_printf("\n");
    }
    auto_printf("final: %5d \n", parent[NUM_VERTICES]);
    auto_printf("\n");
    auto_printf("libs \n");
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
            const int vtx = x + 1 + (BOARD_SIZE + 2) * (y + 1);
            auto_printf("%5d ", libs[vtx]);
        }
        auto_printf("\n");
    }
    auto_printf("final: %5d \n", libs[NUM_VERTICES]);
    auto_printf("\n");
    auto_printf("stones \n");
    for (int y = 0; y < BOARD_SIZE; y++) {
        for (int x = 0; x < BOARD_SIZE; x++) {
           const int vtx = x + 1 + (BOARD_SIZE + 2) * (y + 1);
           auto_printf("%5d ", stones[vtx]);
        }
        auto_printf("\n");
    }
    auto_printf("final: %5d \n", stones[NUM_VERTICES]);
    auto_printf("\n");
}

void Board::display_chain() { m_string.display_chain(); }

std::pair<int, int> Board::get_symmetry(const int x, const int y,
                                        const int symmetry,
                                        const int boardsize) {

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
    assert(symmetry != IDENTITY_SYMMETRY || (x == idx_x && y == idx_y));

    return {idx_x, idx_y};
}

void Board::init_symmetry_table(const int boardsize) {

    for (int sym = 0; sym < NUM_SYMMETRIES; ++sym) {
        for (int vtx = 0; vtx < NUM_VERTICES; ++vtx) {
            symmetry_nn_vtx_table[sym][vtx] = 0;
        }
        for (int idx = 0; idx < NUM_INTERSECTIONS; ++idx) {
            symmetry_nn_idx_table[sym][idx] = 0;
        }
    }

    for (int sym = 0; sym < NUM_SYMMETRIES; ++sym) {
        for (int y = 0; y < boardsize; y++) {
            for (int x = 0; x < boardsize; x++) {
                const auto sym_idx = get_symmetry(x, y, sym, boardsize);
                const auto vtx = get_vertex(x, y);
                const auto idx = get_index(x, y);
                symmetry_nn_idx_table[sym][idx] =
                    get_index(sym_idx.first, sym_idx.second);
                symmetry_nn_vtx_table[sym][vtx] =
                    get_vertex(sym_idx.first, sym_idx.second);
            }
        }
    }
}

void Board::init_dirs(const int boardsize) {
    const int x_shift = boardsize + 2;
    m_dirs[0] = (-x_shift);
    m_dirs[1] = (-1);
    m_dirs[2] = (+1);
    m_dirs[3] = (+x_shift);
    m_dirs[4] = (-x_shift - 1);
    m_dirs[5] = (-x_shift + 1);
    m_dirs[6] = (+x_shift - 1);
    m_dirs[7] = (+x_shift + 1);
}

void Board::init_bitboard(const int numvertices) {
    // m_bitstate[BLACK].init_bitboard(numvertices);
    // m_bitstate[WHITE].init_bitboard(numvertices);
}

void Board::reset_board(const int boardsize, const float komi) {
    set_boardsize(boardsize);
    set_komi(komi);
    reset_board_data(m_boardsize);

    m_komove = NO_VERTEX;
    m_lastmove = NO_VERTEX;
    m_tomove = BLACK;
    m_prisoners = {0, 0};
    m_passes = 0;
    m_movenum = 0;

    init_symmetry_table(m_boardsize);
    init_dirs(m_boardsize);
    //init_bitboard(m_numvertices);

    m_hash = calc_hash(NO_VERTEX);
    m_ko_hash = calc_ko_hash();

    m_string.reset_chain();
}

void Board::set_boardsize(int boardsize) {
    if (boardsize > BOARD_SIZE || boardsize < MARCO_MINIMAL_GTP_BOARD_SIZE) {
        boardsize = BOARD_SIZE;
    }
    m_boardsize = boardsize;
    m_letterboxsize = m_boardsize + 2;
    m_numvertices = m_letterboxsize * m_letterboxsize;
    m_intersections = m_boardsize * m_boardsize;
}

void Board::set_komi(const float komi) {

    const auto old_komi = get_komi();
    m_komi_integer = static_cast<int>(komi);
    m_komi_float = komi - static_cast<float>(m_komi_integer);
    if (m_komi_float < 0.01f && m_komi_float > (-0.01f)) {
        m_komi_float = 0.0f;
    }

    update_zobrist_komi(komi, old_komi);
}

void Board::reset_board_data(const int boardsize) {

    const auto num_vertices = (boardsize+2) * (boardsize+2);
    for (int vtx = 0; vtx < num_vertices; vtx++) {
        m_state[vtx] = INVAL;
        m_neighbours[vtx] = 0;
    }

    m_empty_cnt = 0;

    for (int y = 0; y < boardsize; y++) {
        for (int x = 0; x < boardsize; x++) {
            const int vtx = get_vertex(x, y);
            m_state[vtx] = EMPTY;
            m_empty_idx[vtx] = m_empty_cnt;
            m_empty[m_empty_cnt++] = vtx;

            if (x == 0 || x == (boardsize - 1)) {
                m_neighbours[vtx] += ((1 << BLACK_NBR_SHIFT) |
                                      (1 << WHITE_NBR_SHIFT) |
                                      (1 << EMPTY_NBR_SHIFT));
            } else {
                m_neighbours[vtx] += (2 << EMPTY_NBR_SHIFT);
            }

            if (y == 0 || y == (boardsize - 1)) {
                m_neighbours[vtx] += ((1 << BLACK_NBR_SHIFT) |
                                      (1 << WHITE_NBR_SHIFT) |
                                      (1 << EMPTY_NBR_SHIFT));
            } else {
                m_neighbours[vtx] += (2 << EMPTY_NBR_SHIFT);
            }
        }
    }
}

void Board::set_passes(int val) {

    if (val > 4) {
        val = 4;
     }
     update_zobrist_pass(val, m_passes);
     m_passes = val;
}

void Board::increment_passes() {

    int ori_passes = m_passes;
    m_passes++;
    if (m_passes > 4)
        m_passes = 4;
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

void Board::info_stream(std::ostream &out) const {

    out << "{";
    out << "Next Player : ";
    if (m_tomove == Board::BLACK) {
        out << "Black";
    } else if (m_tomove == Board::WHITE) {
        out << "White";
    } else {
        out << "Error";
    }
    out << ", ";
    out << "Board Size : "  << m_boardsize              << ", ";
    out << "Komi : "        << get_komi()               << ", ";
    // out << "Label Komi : "   << std::setw(2) << cfg_lable_komi           << ", ";
    // out << "Label Buffer : " << std::setw(2) << cfg_label_buffer * 100.f << "%";
    out << "}\n";
}

void Board::hash_stream(std::ostream &out) const {

    out << std::hex;
    out << "HASH : " << m_hash;
    out << " | ";
    out << "KO_HASH : " << m_ko_hash;
    out << "\n";
    out << std::dec;
}

void Board::prisoners_stream(std::ostream &out) const {

    out << "BLACK (X) has captured ";
    out << std::to_string(m_prisoners[BLACK]);
    out << " stones\n";
    out << "WHITE (O) has captured ";
    out << std::to_string(m_prisoners[WHITE]);
    out << " stones\n";
}

void Board::board_stream(std::ostream &out, const int lastmove, bool is_sgf) const {

    m_boardsize > 9 ? (out << spcaces_to_string(3))
                    : (out << spcaces_to_string(2));
    out << columns_to_string(m_boardsize);
    for (int y = 0; y < m_boardsize; y++) {
    
        const int row = (is_sgf ? y : m_boardsize - y - 1);

        out << std::to_string(row + 1);
        if (row < 9 && m_boardsize > 9) {
            out << spcaces_to_string(1);
        }
        if (lastmove == get_vertex(0, row)) {
            out << "(";
        } else {
            out << spcaces_to_string(1);
        }

        for (int x = 0; x < m_boardsize; x++) {
            const int vtx = get_vertex(x, row);
            const auto state = get_state(vtx);
            out << state_to_string(
                       static_cast<vertex_t>(state), is_star(x, row));

            if (lastmove == get_vertex(x, row)) {
                out << ")";
            } else if (x != m_boardsize - 1 && lastmove == get_vertex(x, row) + 1) {
                out << "(";
            } else {
                out << spcaces_to_string(1);
            }
        }
        out << std::to_string(row + 1);
        out << "\n";
    }
    m_boardsize > 9 ? (out << spcaces_to_string(3))
                    : (out << spcaces_to_string(2));
    out << columns_to_string(m_boardsize);
}

void Board::board_stream(std::ostream &out, const int lastmove) const {
    board_stream(out, lastmove, true);
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
    color == BLACK  ? res += "x" : 
    color == WHITE  ? res += "o" : 
    is_star == true ? res += "+" :
    color == EMPTY  ? res += "." : 
    color == INVAL  ? res += "-" : res += "error";
 
    return res;
}

std::string Board::spcaces_to_string(const int times) const {
    auto res = std::string{};
    for (int i = 0; i < times; ++i) {
        res += " ";
    }
    return res;
}

void Board::text_display() const {

    auto out = std::ostringstream{};
    board_stream(out, m_lastmove);
  
    auto res = out.str();
    auto_printf("%s\n", res.c_str());
}

std::uint64_t Board::calc_hash(int komove, int sym) const {

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

std::uint64_t Board::calc_ko_hash(int sym) const {

    std::uint64_t res = Zobrist::zobrist_empty;
    res ^= komi_hash(get_komi());
    for (int vtx = 0; vtx < m_numvertices; vtx++) {
        if (is_on_board(vtx)) {
            res ^= Zobrist::zobrist[m_state[vtx]][get_transform_vtx(vtx, sym)];
        }
    }
    return res;
}

int Board::count_pliberties(const int vtx) const {
  return (m_neighbours[vtx] >> (EMPTY_NBR_SHIFT)) & NBR_MASK;
}

bool Board::is_superko_move(const int vtx,
                            const int color,
                            const std::vector<std::uint64_t> &superko_history) const {

    assert(color == BLACK || color == WHITE);
    assert(vtx != RESIGN && vtx != NO_VERTEX);
    if (vtx == PASS) {
        return false;
    }

    if (m_state[vtx] != EMPTY) {
        return false;
    }
  
    if (count_pliberties(vtx) != 0) {
        return false;
    }

    bool success = false;
    size_t cap_times = 0;

    for (int k = 0; k < 4; ++k) {
        const int avtx = vtx + m_dirs[k];
        const int parent = m_string.parent[avtx];
        const int libs = m_string.libs[parent];
        const int stones = m_string.stones[parent];
        if (libs == 1) {
            cap_times++;
        }
        if (stones == 1 && libs == 1) {
            if (cap_times == 1) {
                success = true;
            } else {
                success = false;
                break;
            }
        } 
    }
    if (success) {
        auto smi_board = std::make_shared<Board>(*this);
        smi_board->play_move(vtx, color);
        auto kohash = smi_board->get_ko_hash();
        auto begin = std::cbegin(superko_history);
        auto end = std::cend(superko_history);
        auto res = std::find(begin, end, kohash);
        return res != end;
    }

    return false;
}

bool Board::is_take_move(const int vtx, const int color) const {
  
    if (m_state[vtx] != EMPTY) {
        return false;
    }

    for (int k = 0; k < 4; ++k) {
        const int avtx = vtx + m_dirs[k];
        const int libs = get_libs(avtx);
    
       if (m_state[avtx] == !color && libs == 1) {
           return true;
       }
    }
    return false;
}

// 判斷是否四周都有棋
bool Board::is_simple_eye(const int vtx, const int color) const {
    return m_neighbours[vtx] & s_eyemask[color];
}

// 確定的真眼
bool Board::is_eye(const int vtx, const int color) const {
  
    if (m_state[vtx] != EMPTY) {
        return false;
    }

    if (!is_simple_eye(vtx, color)) {
        return false;
    }

    int colorcount[4];

    colorcount[BLACK] = 0;
    colorcount[WHITE] = 0;
    colorcount[INVAL] = 0;

    for (int k = 4; k < 8; ++k) {
        const int avtx = vtx + m_dirs[k];
        colorcount[m_state[avtx]]++;
    }

    if (colorcount[INVAL] == 0) {
        if (colorcount[!color] > 1) {
            return false;
        }
    } else {
        if (colorcount[!color] > 0) {
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
        const int acolor = m_state[avtx];
        if (acolor == color && libs > 1) {
            return false;
        } else if (acolor == (!color) && libs <= 1) {
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
    } while (next_pos != aip);

    std::swap(m_string.next[aip], m_string.next[ip]);
}

int Board::remove_string(const int ip) {

    int pos = ip;
    int removed = 0;
    int color = m_state[ip];

    assert(color != EMPTY);

    do {
        remove_stone(pos, color);
        m_string.parent[pos] = NUM_VERTICES;

        m_empty_idx[pos] = m_empty_cnt;
        m_empty[m_empty_cnt] = pos;
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
    int captured_vtx = NO_VERTEX;

    for (int k = 0; k < 4; ++k) {
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

    set_to_move(color);
    const int ori_komove = m_komove;

    if (vtx == PASS) {
        increment_passes();
        m_komove = NO_VERTEX;
    } else {
        if (get_passes() != 0) {
            set_passes(0);
        }
        m_komove = update_board(vtx, color);
    }

    if (m_komove != ori_komove) {
        update_zobrist_ko(m_komove, ori_komove);
    }
    m_lastmove = vtx;
    m_movenum++;

    exchange_to_move();
}

bool Board::is_legal(const int vtx,
                     const int color,
                     Board::avoid_t avoid) const {

    if (vtx == PASS || vtx == RESIGN) {
        return true;
    }

    if (m_state[vtx] != EMPTY) {
        return false;
    }

    if (is_avoid_move(avoid, vtx, color)) {
        return false;
    }

    if (is_suicide(vtx, color)) {
        if (!option<bool>("allow_suicide")) {
            return false;
        }
    }

    if (vtx == m_komove) {
        return false;
    }

    return true;
}

int Board::calc_reach_color(int color) const {

    auto buf = std::vector<bool>(m_numvertices, false);
    auto peekState = [&] (int vtx) {
        return m_state[vtx];
    };

    return calc_reach_color(color, EMPTY, buf, peekState);
}

int Board::collect_group(const int spread_center,
                         const int spread_color,
                         std::vector<bool>& buf,
                         std::function<int(int)> f_peek) const {

    if (buf.size() != (size_t)m_numvertices) {
        buf.resize(m_numvertices);
    }

    int reachable = 0;
    auto open = std::queue<int>();
    auto color = f_peek(spread_center);

    if (color == spread_color) {
        open.push(spread_center);
        buf[spread_center] = true;
        reachable++;
    }

    while (!open.empty()) {
        const auto vtx = open.front();
        open.pop();

        for (int k = 0; k < 4; ++k) {
            const auto neighbor = vtx + m_dirs[k];
            const auto peek = f_peek(neighbor);
            if (!buf[neighbor] && peek == spread_color) {
                reachable++;
                buf[neighbor] = true;
                open.emplace(neighbor);
            }
       }
    }

    return reachable;
}


int Board::calc_reach_color(int color, int spread_color,
                            std::vector<bool> &buf, std::function<int(int)> f_peek) const {

    if (buf.size() != (size_t)m_numvertices) {
        buf.resize(m_numvertices);
    }

    int reachable = 0;
    auto open = std::queue<int>();
    for (int y = 0; y < m_boardsize; ++y) {
        for (int x = 0; x < m_boardsize; ++x) {
            const auto vertex = get_vertex(x, y);
            const auto peek = f_peek(vertex);

            if (peek == color) {
                reachable++;
                buf[vertex] = true;
                open.emplace(vertex);
            } else {
                buf[vertex] = false;
            }
        }
    }
    while (!open.empty()) {
        const auto vertex = open.front();
        open.pop();

        for (int k = 0; k < 4; ++k) {
            const auto neighbor = vertex + m_dirs[k];
            const auto peek = f_peek(neighbor);

            if (!buf[neighbor] && peek == spread_color) {
                reachable++;
                buf[neighbor] = true;
                open.emplace(neighbor);
            }
        }
    }
    return reachable;
}

float Board::area_score(float komi, Board::rule_t rule) const {

    if (rule == rule_t::Tromp_Taylor) {

        const auto white = calc_reach_color(WHITE);
        const auto black = calc_reach_color(BLACK);
        return static_cast<float>(black - white) - komi;

    } else if (rule == rule_t::Jappanese) {

        const auto terri = compute_territory();
        const auto white = terri.second + m_prisoners[WHITE];
        const auto black = terri.first + m_prisoners[BLACK];

        return static_cast<float>(black - white) - komi;
    }

    return 0.0f;
}

int Board::area_distance() const {

    const auto white = calc_reach_color(WHITE);
    const auto black = calc_reach_color(BLACK);
    return black - white;
}

void Board::find_dame(std::array<territory_t, NUM_VERTICES> &territory) const {
    auto black = std::vector<bool>(m_numvertices, false);
    auto white = std::vector<bool>(m_numvertices, false);

    auto peekState = [&] (int vtx) {
        return m_state[vtx];
    };

    calc_reach_color(BLACK, EMPTY, black, peekState);
    calc_reach_color(WHITE, EMPTY, white, peekState);

    for (int y = 0; y < m_boardsize; ++y) {
        for (int x = 0; x < m_boardsize; ++x) {
            const auto vertex = get_vertex(x, y);
            if (black[vertex] && white[vertex]) {
                territory[vertex] = DAME;
            }
        }
    }
}

void Board::find_seki(std::array<territory_t, NUM_VERTICES> &territory) const {
    auto black_seki = std::vector<bool>(m_numvertices, false);
    auto white_seki = std::vector<bool>(m_numvertices, false);

    auto peekTerritory = [&] (int vtx) {
        return territory[vtx];
    };

    calc_reach_color(DAME, B_STONE, black_seki, peekTerritory);
    calc_reach_color(DAME, W_STONE, white_seki, peekTerritory);

    for (int y = 0; y < m_boardsize; ++y) {
        for (int x = 0; x < m_boardsize; ++x) {
            const auto vertex = get_vertex(x, y);
            if ((black_seki[vertex] || white_seki[vertex]) &&
                territory[vertex] != DAME) {
                territory[vertex] = SEKI;
            }
        }
    }
}

std::pair<int, int> Board::find_territory(std::array<territory_t, NUM_VERTICES> &territory) const {
    int b_terr_count = 0;
    int w_terr_count = 0;

    auto seki_eye = std::vector<bool>(m_numvertices, false);
    auto b_territory = std::vector<bool>(m_numvertices, false);
    auto w_territory = std::vector<bool>(m_numvertices, false);

    auto peekTerritory = [&] (int vtx) {
        return territory[vtx];
    };
    calc_reach_color(SEKI, EMPTY_I, seki_eye, peekTerritory);
    calc_reach_color(B_STONE, EMPTY_I, b_territory, peekTerritory);
    calc_reach_color(W_STONE, EMPTY_I, w_territory, peekTerritory);

    for (int y = 0; y < m_boardsize; ++y) {
        for (int x = 0; x < m_boardsize; ++x) {
            const auto vertex = get_vertex(x, y);
            if (seki_eye[vertex] && territory[vertex] != SEKI) {
                territory[vertex] = SEKI_EYE;
            } else if (b_territory[vertex] && territory[vertex] != B_STONE) {
                territory[vertex] = B_TERR;
                b_terr_count++;
            } else if (w_territory[vertex] && territory[vertex] != W_STONE) {
                territory[vertex] = W_TERR;
                w_terr_count++;
            }
        }
    }
    return std::make_pair(b_terr_count, w_terr_count);
}

std::pair<int, int> Board::compute_territory() const {

    auto territory = std::array<territory_t, NUM_VERTICES>{};
    reset_territory(territory);
    find_dame(territory);
    find_seki(territory);
    return find_territory(territory);
}

void Board::reset_territory(std::array<territory_t, NUM_VERTICES> &territory) const {
    for (int vtx = 0; vtx < m_numvertices; ++vtx) {
        switch (m_state[vtx]) {
            case BLACK:
                territory[vtx] = B_STONE;
                break;
            case WHITE:
                territory[vtx] = W_STONE;
                break;
            case EMPTY:
                territory[vtx] = EMPTY_I;
                break;
            case INVAL:
                territory[vtx] = INVAL_I;
                break;
        }
    }
}

bool Board::is_avoid_move(Board::avoid_t avoid,
                          const int vtx,
                          const int color) const {

    if (avoid == avoid_t::NONE) {
        return false;
    } else if (avoid == avoid_t::SEARCH_BLOCK) {
        auto eye = is_eye(vtx, color);
        if (eye) {
            auto eyes_collect = std::vector<int>{};
            auto ownership = get_ownership();
            auto group = std::vector<bool>(m_numvertices, false);

            const auto peekState = [&] (int vtx) {
                if (is_on_board(vtx)) {
                    const auto x = get_x(vtx);
                    const auto y = get_y(vtx);
                    const auto index = get_index(x, y);
                    return ownership[index];
                }
                return static_cast<int>(INVAL);
            };

            // 在同一快地裡就被視為同一個棋塊
            const auto res = collect_group(vtx, color, group, peekState);
            assert(res != 0);
      
            // 搜尋棋塊內所有的真眼，避免將所有真眼都填掉
            for (int y = 0; y < m_boardsize; ++y) {
                for (int x = 0; x < m_boardsize; ++x) {
                    const auto vertex = get_vertex(x, y);
                    if (group[vertex] && is_eye(vertex, color)) {
                        auto opc = size_t{0};
                        auto outside = size_t{0};
                        for (int k = 4; k < 8; ++k) {
                            const auto avtx = vertex + m_dirs[k];
                            if (m_state[avtx] == INVAL) {
                                outside++;
                            } 

                            if (group[avtx]) {
                                opc++;
                            }
                        }
                        if (opc >= 3 || opc + outside == 4) {
                            eyes_collect.emplace_back(vertex);
                        }
                    }
                }
            }
            bool alive_group = eyes_collect.size() >= 2;
            return alive_group;
        }
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

int Board::get_boardsize() const {
    return m_boardsize;
}

int Board::get_intersections() const {
    return m_intersections;
}

int Board::get_komi_integer() const {
    return m_komi_integer;
}

float Board::get_komi_float() const {
    return m_komi_float;
}

float Board::get_komi() const {
    return m_komi_float + static_cast<float>(m_komi_integer);
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

int Board::get_komove() const {
    return m_komove;
}

int Board::get_stones(const int x, const int y) const {
    return get_stones(get_vertex(x, y));
}

int Board::get_stones(const int vtx) const {
    return m_string.stones[m_string.parent[vtx]];
}

int Board::get_libs(const int x, const int y) const {
    return get_libs(get_vertex(x, y));
}

int Board::get_libs(const int vtx) const {
    return m_string.libs[m_string.parent[vtx]];
}

std::vector<int> Board::get_ownership() const {

    auto res = std::vector<int>(m_intersections, INVAL);

    auto black = std::vector<bool>(m_numvertices, false);
    auto white = std::vector<bool>(m_numvertices, false);
    const auto peekState = [&] (const int vtx) {
        return m_state[vtx];
    };

    calc_reach_color(BLACK, EMPTY, black, peekState);
    calc_reach_color(WHITE, EMPTY, white, peekState);

    for (int y = 0; y < m_boardsize; ++y) {
        for (int x = 0; x < m_boardsize; ++x) {
            const auto idx = get_index(x, y);
            const auto vtx = get_vertex(x, y);
            if ((black[vtx] && white[vtx]) || (!black[vtx] && !white[vtx])) {
                //if there are no stones on board, this is seki case
                res[idx] = EMPTY;
            } else if (black[vtx]) {
                res[idx] = BLACK;  
            } else if (white[vtx]) {
                res[idx] = WHITE;
            }
       }
    }

    return res;
}


int Board::get_groups(std::vector<int> &groups) const {

    groups = std::vector<int>(m_numvertices, 0);
    int cnt = 0;
    auto ownership = get_ownership();

    const auto peekState = [&] (int vtx) {
        if (is_on_board(vtx)) {
            const auto x = get_x(vtx);
            const auto y = get_y(vtx);
            const auto index = get_index(x, y);
            return ownership[index];
        }
        return static_cast<int>(INVAL);
    };

    for (int y = 0; y < m_boardsize; ++y) {
        for (int x = 0; x < m_boardsize; ++x) {
            const auto vtx = get_vertex(x, y);
            const auto color = m_state[vtx];
            if (color != EMPTY && groups[vtx] == 0) {
                cnt++;
                auto group = std::vector<bool>(m_numvertices, false);

                // 在同一快地裡就被視為同一個棋塊
                const auto res = collect_group(vtx, color, group, peekState);

                assert(res != 0);

                for (int v = 0; v < m_numvertices; ++v) {
                    if (group[v]) {
                        groups[v] = cnt;
                    }
                }
            }
        }
    }

    return cnt;
}

int Board::get_alive_groups(std::vector<int> &alive_groups) const {
    alive_groups = std::vector<int>(m_numvertices, 0);
    auto groups = std::vector<int>{};
    int alive_cnt = 0;

    const auto cnt = get_groups(groups);
    const auto ownership = get_ownership();

    for (int c = 1; c <= cnt; ++c) {
        auto eyes_collect = std::vector<int>{};

        for (int y = 0; y < m_boardsize; ++y) {
            for (int x = 0; x < m_boardsize; ++x) {
                const auto vtx = get_vertex(x, y);
                const auto idx = get_index(x, y);
                const auto color = ownership[idx];
                if (groups[vtx] == c && is_eye(vtx, color)) {
                    auto opc = size_t{0};
                    auto outside = size_t{0};
                    for (int k = 4; k < 8; ++k) {
                        const auto avtx = vtx + m_dirs[k];
                        if (m_state[avtx] == INVAL) {
                            outside++;
                        } 

                        if (groups[avtx] == c) {
                            opc++;
                        }
                    }
                    if (opc >= 3 || opc + outside == 4) {
                        eyes_collect.emplace_back(vtx);
                    }  
                }
            }
        }
        if (eyes_collect.size() >= 2) {
            alive_cnt++;
            for (int v = 0; v < m_numvertices; ++v) {
                if (groups[v] == c) {
                    alive_groups[v] = alive_cnt;
                }
            }
        }
    }

    return alive_cnt;
}

std::vector<int> Board::get_alive_seki(const int color) const {
  
    auto alive_seki = std::vector<int>{};

    auto alive_groups  = std::vector<int>{};
    get_alive_groups(alive_groups);

    const auto ownership = get_ownership();
    for (int y = 0; y < m_boardsize; ++y) {
        for (int x = 0; x < m_boardsize; ++x) {
            const auto vtx = get_vertex(x, y);
            const auto idx = get_index(x, y);
            const auto owner = ownership[idx];

            if (owner == EMPTY) {
                // 如果單官太空，我們會認為此空尚未確定
                int cnt = 0;
                for (int k = 0; k < 4; ++k) {
                    const auto avtx = vtx + m_dirs[k];
                    if (m_state[avtx] == EMPTY || m_state[avtx] == INVAL) {
                        cnt++;
                    }
                }
                if (cnt == 4) {
                    alive_seki.emplace_back(vtx);
                    continue;
                }

                // 如果單官連接活棋，代表此空尚未確定
                for (int k = 0; k < 4; ++k) {
                    const auto avtx = vtx + m_dirs[k];
                    if (is_on_board(avtx)) {
                        const auto x = get_x(avtx);
                        const auto y = get_y(avtx);
                        const auto aidx = get_index(x, y);
                        if(alive_groups[avtx] != 0 && ownership[aidx] == color) {
                            alive_seki.emplace_back(vtx);
                            break;
                        }
                    }
                }
            }
        }
    }

    return alive_seki;
}

void Board::vertex_stream(std::ostream &out, const int vertex) const {

    assert(vertex != NO_VERTEX);

    if (vertex == PASS) {
       out << "pass";
       return;
    } else if (vertex == RESIGN) {
       out << "resign";
       return;
    }
    const auto x = get_x(vertex);
    const auto y = get_y(vertex);
    auto x_char = static_cast<char>(x + 65);
    if (x_char >= 'I') {
        x_char++;
    }
    auto y_str = std::to_string(y + 1);

    out << x_char;
    out << y_str;
}

std::string Board::vertex_to_string(const int vertex) const {
    auto res = std::ostringstream{};
    vertex_stream(res, vertex);
    return res.str();
}

void Board::sgf_stream(std::ostream &out,
                      const int vertex, const int color) const {
    assert(vertex != NO_VERTEX);
  
    if (color == BLACK) {
        out << "B";
    } else if (color == WHITE) {
        out << "W";
    } else {
        out << "error";
    }

    if (vertex == PASS || vertex == RESIGN) {
        out << "[]";
    } else if (vertex > NO_VERTEX && vertex < m_numvertices){
        const auto x = get_x(vertex);
        const auto y = get_y(vertex);

        const auto x_char = static_cast<char>(x +
                                (x < 25 ? 'a' : 'A'));
        const auto y_char = static_cast<char>(y +
                                (y < 25 ? 'a' : 'A'));

        out << '[';
        out << x_char << y_char;
        out << ']';
    } else {
        out << "[error]";
    }
}

void Board::sgf_stream(std::ostream &out) const {
    sgf_stream(out, m_lastmove, !(m_tomove));
}

// Find the liberties of string.
int Board::find_liberties(const int vtx,
                          std::vector<int>& buf) const {
    auto num_found = size_t{0};
    auto next = vtx;
    do {
        for(int k = 0; k < 4; ++k) {
            const auto avtx = next + m_dirs[k];
            if(m_state[avtx] == EMPTY) {
                auto begin = std::begin(buf);
                auto end = std::end(buf);
                auto res = std::find(begin, end, avtx);
                if (res == end) {
                    buf.emplace_back(avtx);
                    num_found++;
                }
            }
        }
        next = m_string.next[next];
    } while (next != vtx);

    return num_found;
}

int Board::find_liberty_gaining_captures(const int vtx,
                                         std::vector<int>& buf) const {
    const int color = m_state[vtx];
    const int opp = !(color);

    assert(color == BLACK || color == WHITE);

    auto strings_buf = std::vector<int>{};
    int num_found = 0;
    int next = vtx;

    do {
        for(int k = 0; k < 4; ++k) {
            const int avtx = next + m_dirs[k];
            if(m_state[avtx] == opp) {
                const int aip = m_string.parent[avtx];
                if(m_string.libs[aip] == 1) {
                    auto begin = std::begin(strings_buf);
                    auto end = std::end(strings_buf);
                    auto res = std::find(begin, end, avtx);
                    if (res == end) {
                        num_found += find_liberties(avtx, buf);
                    } else {
                        strings_buf.emplace_back(avtx);
                    }
                }
            }
        }
        next = m_string.next[next];
    } while (next != vtx);

    return num_found;
}


// Get the possible lowest and most liberties. 
std::pair<int, int> Board::get_libs_for_ladder(const int vtx, const int color) const {

    const int stone_libs = count_pliberties(vtx);
    const int opp = (!color);

    int num_caps = 0; //number of adjacent directions in which we will capture
    int potential_libs_from_Caps = 0; //Total number of stones we're capturing (possibly with multiplicity)
    int num_connection_libs = 0; //Sum over friendly groups connected to of their libs-1
    int max_connection_libs = stone_libs; //Max over friendly groups connected to of their libs-1
  
    for (int k = 0; k < 4; ++k) {
        const auto avtx = vtx + m_dirs[k];
        const auto acolor = m_state[avtx];

        if (acolor == color) {
            const int aparent = m_string.parent[avtx];
            const int alibs = m_string.libs[aparent] - 1;
            num_connection_libs += alibs;

            if(alibs > max_connection_libs) {
                max_connection_libs = alibs; 
            }
        } else if (acolor == opp) {
            const int aparent = m_string.parent[avtx];
            const int alibs = m_string.libs[aparent];
            if (alibs == 1) {
                num_caps++;
                potential_libs_from_Caps += m_string.stones[aparent];
            }
        }
    }
    const int lower_bound =
        num_caps + max_connection_libs; 
    const int upper_bound = 
        stone_libs + potential_libs_from_Caps + num_connection_libs;

    return std::pair<int, int>{lower_bound, upper_bound};
}

Board::ladder_t Board::prey_selections(const int prey_color,
                                       const int ladder_vtx,
                                       std::vector<int>& selections, bool think_ko) const {
    assert(selections.empty());

    const int libs = get_libs(ladder_vtx);
    if (libs >= 2 || (m_komove != NO_VERTEX && think_ko)) {
        // If we are the prey and the hunter left a simple ko point, assume we already win
        // because we don't want to say yes on ladders that depend on kos
        // This should also hopefully prevent any possible infinite loops - I don't know of any infinite loop
        // that would come up in a continuous atari sequence that doesn't ever leave a simple ko point.

        return ladder_t::GOOD_FOR_PREY;
    }
    assert(libs == 1);

    int num_move = find_liberties(ladder_vtx, selections);
    assert(num_move == libs);
    const int move = selections[0];

    num_move += find_liberty_gaining_captures(ladder_vtx, selections);

    // Must be a legal move.
    selections.erase(
        std::remove_if(std::begin(selections), std::end(selections), [=](int v){ return !is_legal(v, prey_color); }),
        std::end(selections)
    );

    // If there is no legal move, the ladder string must die.
    num_move = selections.size();
    if (num_move == 0) {
        return ladder_t::GOOD_FOR_HUNTER; 
    }

    if (selections[0] == move) {
        auto bound = get_libs_for_ladder(move, prey_color);
        const auto lower_bound = bound.first;
        const auto upper_bound = bound.second;
        if (lower_bound >= 3) {
            return ladder_t::GOOD_FOR_PREY;
        }
        if (num_move == 1  && upper_bound == 1) {
            return ladder_t::GOOD_FOR_HUNTER;
        }
    }

    return ladder_t::GOOD_FOR_NONE; // keep running
}

bool Board::is_neighbor(const int vtx_1, const int vtx_2) const {
    for (int k = 0; k < 4; ++k) {
        if ((vtx_1 + m_dirs[k]) == vtx_2) {
             return true;
        }
    }
    return false;
}

Board::ladder_t Board::hunter_selections(const int prey_color,
                                         const int ladder_vtx, std::vector<int>& selections) const {
    assert(selections.empty());

    const int libs = get_libs(ladder_vtx);

    if (libs >= 3) {
        return ladder_t::GOOD_FOR_PREY;
    }
    else if (libs <= 1) {
        return ladder_t::GOOD_FOR_HUNTER;
    }
  
    assert(libs == 2);

    auto buf = std::vector<int>{};
    int num_libs = find_liberties(ladder_vtx, buf);

    assert(num_libs == libs);
    const int move_1 = buf[0];
    const int move_2 = buf[1];
    //TODO: avoid double-ko death (kata go)

    if (!is_neighbor(move_1, move_2)) {
        size_t size = 0;
        const int hunter_color = (!prey_color);
        const int libs_1 = count_pliberties(move_1); 
        const int libs_2 = count_pliberties(move_2); 

        if (libs_1 >= 3 && libs_2 >= 3) {
            return ladder_t::GOOD_FOR_PREY;
        }
        else if (libs_1 >= 3) {
            if (is_legal(move_1, hunter_color)) {
                selections.emplace_back(move_1);
                size++;
            }
        }
        else if (libs_2 >= 3) {
            if (is_legal(move_2, hunter_color)) {
                selections.emplace_back(move_2);
                size++;
            }
        } else {
            if (is_legal(move_1, hunter_color)) {
                selections.emplace_back(move_1);
                size++;
            }
            if (is_legal(move_2, hunter_color)) {
                selections.emplace_back(move_2);
                size++;
            }
        }
    }

    if (selections.size() == 0) {
        return ladder_t::GOOD_FOR_PREY;
    }

    return ladder_t::GOOD_FOR_NONE; // keep running
}

Board::ladder_t Board::hunter_move(std::shared_ptr<Board> board,
                                   const int prey_vtx, const int prey_color,
                                   const int ladder_vtx, size_t& ladder_nodes, bool fork) const {
    if ((++ladder_nodes) >= MAX_LADDER_NODES) {
        // If hit the limit, assume prey is winner. 
        return ladder_t::GOOD_FOR_PREY;
    }

    std::shared_ptr<Board> ladder_board;
    if (fork) {
        ladder_board = std::make_shared<Board>(*board);
    } else {
        ladder_board = board;
    }

    if (prey_vtx != NO_VERTEX) {
        // Prey move first.
        ladder_board->play_move(prey_vtx, prey_color);
    }
    ladder_board->set_to_move(!prey_color);

    // Search possible move(s) for hunter.
    auto selections = std::vector<int>{};
    auto res = ladder_board->hunter_selections(prey_color, ladder_vtx, selections);

    if (res != ladder_t::GOOD_FOR_NONE) {
        return res;
    }
  
    bool next_fork = true;
    const auto selection_size = selections.size();
    if (selection_size == 1) {
        next_fork = false;
    }

    auto best = ladder_t::GOOD_FOR_NONE;
 
    for (auto i = size_t{0}; i < selection_size; ++i) {
        const int vtx = selections[i];
        auto next_res = prey_move(ladder_board, vtx, 
                                  prey_color, ladder_vtx, 
                                  ladder_nodes, next_fork);

        assert(next_res != ladder_t::GOOD_FOR_NONE);

        best = next_res;
        if (next_res == ladder_t::GOOD_FOR_HUNTER) {
            break;
        }
    }

    return best;
}

Board::ladder_t Board::prey_move(std::shared_ptr<Board> board,
                                 const int hunter_vtx, const int prey_color,
                                 const int ladder_vtx, size_t& ladder_nodes, bool fork) const {

    if ((++ladder_nodes) >= MAX_LADDER_NODES) {
        // If hit the limit, assume prey is winner. 
        return ladder_t::GOOD_FOR_PREY;
    }

    std::shared_ptr<Board> ladder_board;
    if (fork) {
        ladder_board = std::make_shared<Board>(*board);
    } else {
        ladder_board = board;
    }

    if (hunter_vtx != NO_VERTEX) {
        // Hunter move first.
        ladder_board->play_move(hunter_vtx, !prey_color);
    }
    ladder_board->set_to_move(prey_color);

    // Search possible move(s) for prey.
    auto selections = std::vector<int>{};
    auto res = ladder_board->prey_selections(prey_color, ladder_vtx, selections, hunter_vtx != NO_VERTEX);

    if (res != ladder_t::GOOD_FOR_NONE) {
        return res;
    }

    bool next_fork = true;
    const size_t selection_size = selections.size();
    if (selection_size == 1) {
        next_fork = false;
    }

    auto best = ladder_t::GOOD_FOR_NONE;

    for (auto i = size_t{0}; i < selection_size; ++i) {
        const int vtx = selections[i];
        auto next_res = hunter_move(ladder_board, vtx,
                                    prey_color, ladder_vtx,
                                    ladder_nodes, next_fork);

        assert(next_res != ladder_t::GOOD_FOR_NONE);

        best = next_res;
        if (next_res == ladder_t::GOOD_FOR_PREY) {
            break;
        }
    }

    return best;
}

bool Board::is_ladder(const int vtx) const {

    if (vtx == PASS) {
        return false;
    }

    const int prey_color = m_state[vtx];
    if (prey_color == EMPTY || prey_color == INVAL) {
        return false;
    }
    // const int hunter_color = !prey_color;

    const int libs = get_libs(vtx);
    const int ladder_vtx = vtx; 
    size_t search_nodes = 0;
    auto res = ladder_t::GOOD_FOR_NONE;
    if (libs == 1) {
        auto ladder_board = std::make_shared<Board>(*this);
        res = prey_move(ladder_board,
                        NO_VERTEX, prey_color,
                        ladder_vtx, search_nodes, false);
    } else if (libs == 2) {
        auto ladder_board = std::make_shared<Board>(*this);
        res = hunter_move(ladder_board,
                          NO_VERTEX, prey_color,
                          ladder_vtx, search_nodes, false);

    } else if (libs >= 3) {
        res = ladder_t::GOOD_FOR_PREY;
    }

    assert(res != ladder_t::GOOD_FOR_NONE);
    return res == ladder_t::GOOD_FOR_HUNTER;
}

std::vector<Board::ladder_t> Board::get_ladders() const {

    auto res = std::vector<Board::ladder_t>(m_intersections, ladder_t::NOT_LADDER);
    auto checked = std::vector<int>{};

    for (int y = 0; y < m_boardsize; ++y) {
        for (int x = 0; x < m_boardsize; ++x) {
            const auto idx = get_index(x, y);
            const auto vtx = get_vertex(x, y);

            int libs = 0;

            auto begin = std::begin(checked);
            auto end = std::end(checked);
            auto parent = m_string.parent[vtx];
            auto found = std::find(begin, end, parent);
            if (found != end) {
                // Found!
                libs = get_libs(vtx);

            } else {
                // Not found!
                const auto ladder = is_ladder(vtx);
                if (ladder) {
                    checked.emplace_back(parent);
                    libs = get_libs(vtx);
                }
            }

            if (libs == 0) {
                continue;
            }

            assert(libs == 1 || libs == 2);
            if (libs == 1) {
                // The ladder string is already death.
                res[idx] = ladder_t::LADDER_DEATH;
            } else {
                // The ladder string has a chance to escape.
                res[idx] = ladder_t::LADDER_ESCAPABLE;
            }

            if (found == end) {
                auto buf = std::vector<int>{};
                auto num_move = find_liberties(vtx, buf);
                assert(num_move == libs);

                for (const auto &v : buf) {
                    const auto ax = get_x(v);
                    const auto ay = get_y(v);
                    const auto aidx = get_index(ax, ay); 
                    if (libs == 1) {
                        res[aidx] = ladder_t::LADDER_TAKE;
                    } else {
                        res[aidx] = ladder_t::LADDER_ATARI;
                    }
                }
            }
        }
    }


    return res;
}

