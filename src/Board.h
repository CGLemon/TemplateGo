#ifndef BOARD_H_INCLUDE
#define BOARD_H_INCLUDE

#include <array>
#include <cassert>
#include <cstdlib>
#include <memory>
#include <string>

#include "Zobrist.h"
#include "config.h"

#define BLACK_NUMBER (0)
#define WHITE_NUMBER (1)
#define EMPTY_NUMBER (2)
#define INVAL_NUMBER (3)

#define NBR_SHIFT (4)
#define BLACK_NBR_SHIFT (BLACK_NUMBER * 4)
#define WHITE_NBR_SHIFT (WHITE_NUMBER * 4)
#define EMPTY_NBR_SHIFT (EMPTY_NUMBER * 4)

#define NBR_MASK (0xf)
#define BLACK_EYE_MASK (4 * (1 << BLACK_NBR_SHIFT))
#define WHITE_EYE_MASK (4 * (1 << WHITE_NBR_SHIFT))

static std::array<int, 2> s_eyemask = {BLACK_EYE_MASK, WHITE_EYE_MASK};

/*
/ TODO: 增加征子的判斷（ 為了支援 KataGo ）
*/

class Board {
public:
  enum class rule_t { Tromp_Taylor, Jappanese };

  enum class avoid_t { NONE, REAL_EYE };

  // Ladder helper
  enum class ladder_t {
    GOOD_FOR_HUNTER,
    GOOD_FOR_PREY,
    GOOD_FOR_NONE,

  };
  static constexpr size_t MAX_LADDER_NODES = 1000;

  enum vertex_t : std::uint8_t {
    BLACK = BLACK_NUMBER,
    WHITE = WHITE_NUMBER,
    EMPTY = EMPTY_NUMBER,
    INVAL = INVAL_NUMBER
  };

  enum territory_t : std::uint8_t {
    B_STONE = 0,
    W_STONE = 1,
    EMPTY_I = 2,
    INVAL_I = 3,
    DAME = 4,
    SEKI = 5,
    SEKI_EYE = 6,
    W_TERR = 7,
    B_TERR = 8
  };


  static constexpr int NO_VERTEX = 0;
  static constexpr int PASS = NUM_VERTICES + 1;
  static constexpr int RESIGN = NUM_VERTICES + 2;

  static constexpr int NUM_SYMMETRIES = 8;
  static constexpr int IDENTITY_SYMMETRY = 0;

  static std::array<std::array<int, NUM_INTERSECTIONS>, NUM_SYMMETRIES>
      symmetry_nn_idx_table;
  static std::array<std::array<int, NUM_VERTICES>, NUM_SYMMETRIES>
      symmetry_nn_vtx_table;
  static std::array<int, 8> m_dirs;

  void reset_board(const int boardsize, const float komi);
  void set_komi(const float komi);
  void set_boardsize(int boardsize);
  void reset_board_data();

  void set_state(const int vtx, const vertex_t);
  void set_to_move(int color);
  void exchange_to_move();

  /*
  / =====================================================================
  / 顯示目前的盤面
  */
  bool is_star(const int x, const int y) const;
  std::string board_to_string(const int lastmove = NO_VERTEX) const;
  std::string state_to_string(const vertex_t color, bool is_star) const;
  std::string spcaces_to_string(const int times) const;
  std::string columns_to_string(const int bsize) const;
  std::string prisoners_to_string() const;
  std::string hash_to_string() const;
  std::string to_move_to_string() const;
  void text_display();
  void display_chain();
  /*
    =====================================================================
  */

  /*
  / =====================================================================
  / 獲取棋盤資訊
  */
  std::pair<int, int> get_symmetry(const int x, const int y, const int symmetry,
                                   const int boardsize);
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

  int get_libs(const int x, const int y) const;
  int get_libs(const int vtx) const;
  int get_komove() const;

  std::uint64_t get_ko_hash() const;
  std::uint64_t get_hash() const;
  /*
    =====================================================================
  */

  std::uint64_t calc_hash(int komove, int sym = IDENTITY_SYMMETRY);
  std::uint64_t calc_ko_hash(int sym = IDENTITY_SYMMETRY);

  void set_passes(int val);
  void increment_passes();

  /*
  / =====================================================================
  / 更新雜湊值
  */
  void update_zobrist(const int vtx, const int new_color, const int old_color);
  void update_zobrist_pris(const int color, const int new_pris,
                           const int old_pris);
  void update_zobrist_tomove(const int new_color, const int old_color);
  void update_zobrist_ko(const int new_komove, const int old_komove);
  void update_zobrist_pass(const int new_pass, const int old_pass);
  /*
    =====================================================================
  */

  int get_x(const int vtx) const;
  int get_y(const int vtx) const;
  std::pair<int, int> get_xy(const int vtx) const;

  int count_pliberties(const int vtx) const;
  bool is_superko_move(const int vtx, const int color, 
                       std::uint64_t *super_ko_history) const;
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
                std::uint64_t *superko_history = nullptr,
                avoid_t avoid = avoid_t::NONE) const;
  bool is_avoid_to_move(avoid_t, const int vtx, const int color) const;

  int calc_reach_color(int color) const;
  int calc_reach_color(int color, int spread_color, std::vector<bool> &bd,
                       bool is_territory) const;
  float area_score(float komi, rule_t = rule_t::Tromp_Taylor);

  std::string vertex_to_string(int vertex) const;


  /*
  / TODO: seki 在非常特殊的情況下會搜尋失敗
  */
  void find_dame();
  void find_seki();
  void reset_territory();
  std::pair<int, int> find_territory();
  std::pair<int, int> compute_territory();
  
  // Ladder helper
  bool is_neighbor(const int vtx_1, const int vtx_2) const;
  bool would_be_ko_move(const int vtx, const int color) const;
  int find_liberties(const int vtx, std::vector<int>& buf) const;
  int find_liberty_gaining_captures(const int vtx, std::vector<int>& buf) const;
  std::pair<int, int> get_libs_for_ladder(const int vtx, const int color) const;

  ladder_t prey_selections(const int prey_color, const int parent, std::vector<int>& selection) const;
  ladder_t hunter_selections(const int prey_color, const int parent, std::vector<int>& selection) const;
  // 計算被征子方能否逃脫
  ladder_t hunter_move(std::shared_ptr<Board> board,
                       const int prey_color, const int parent,
                       size_t& ladder_nodes, bool fork) const;
  ladder_t prey_move(std::shared_ptr<Board> board,
                     const int prey_color, const int parent,
                     size_t& ladder_nodes, bool fork) const;
  bool is_ladder(const int vtx) const;

private:
  /*
  / TODO: BitBoard 導入 patten 並快速比對
  */
  class BitBoard {
  public:
    BitBoard() = default;
    BitBoard(const int numvertices) { init_bitboard(numvertices); }
    void init_bitboard(const int numvertices);
    void add_bit(const int vertex);
    void remove_bit(const int vertex);

    std::vector<std::uint64_t> bitboard;

  private:
    std::uint64_t make_bit(const int idx);
  };
  /*
  / Chain: 棋串的資料結構
  /        棋串由單個棋子組成
  */
  struct Chain {
    std::array<std::uint16_t, NUM_VERTICES+1> next; // next:   棋子下一個連接的棋子
    std::array<std::uint16_t, NUM_VERTICES+1> parent; // parent: 棋串的編號，同一個棋串編號相同
    std::array<std::uint16_t, NUM_VERTICES+1> libs; // libs:   棋串的氣
    std::array<std::uint16_t, NUM_VERTICES+1> stones; // stones: 棋串包含的棋子數目
    void reset_chain();
    void add_stone(const int vtx, const int lib);
    void display_chain();
  };

  std::array<BitBoard, 2> m_bitstate; // 比特棋盤，目前無用
  std::array<vertex_t, NUM_VERTICES> m_state; // 棋盤狀態，包含黑子、白子、空白
  std::array<std::uint16_t, NUM_VERTICES> m_neighbours; // 四周的黑棋個數、白棋個數、空白個數，Leela Zero 獨有的資料型態
  std::array<std::uint16_t, NUM_VERTICES> m_empty;
  std::array<std::uint16_t, NUM_VERTICES> m_empty_idx;

  std::array<territory_t, NUM_VERTICES> m_territory; // 用於日式計地規則
  Chain m_string;                                    // 棋串

  std::array<int, 2> m_prisoners; // 提子數

  std::uint64_t m_hash; // 雜湊值，考慮棋盤狀態、提子數、下一手為黑棋或白棋、打劫和虛手數目
  std::uint64_t m_ko_hash; // 雜湊值，僅考慮棋盤狀態，為了避免相同盤面產生

  int m_tomove; // 下一手為黑棋或白棋
  int m_letterboxsize;
  int m_numvertices;
  int m_intersections;
  int m_boardsize; // 棋盤大小
  int m_lastmove;  // 上一手下過的位置
  int m_komove;    // 打劫的位置
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
  return (y + 1) * m_letterboxsize + (x + 1);
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

inline void Board::update_zobrist(const int vtx, const int new_color,
                                  const int old_color) {
  m_hash ^= Zobrist::zobrist[old_color][vtx];
  m_hash ^= Zobrist::zobrist[new_color][vtx];
  m_ko_hash ^= Zobrist::zobrist[old_color][vtx];
  m_ko_hash ^= Zobrist::zobrist[new_color][vtx];
}

inline void Board::update_zobrist_pris(const int color, const int new_pris,
                                       const int old_pris) {
  m_hash ^= Zobrist::zobrist_pris[color][old_pris];
  m_hash ^= Zobrist::zobrist_pris[color][new_pris];
}

inline void Board::update_zobrist_tomove(const int new_color,
                                         const int old_color) {
  if (old_color != new_color) {
    m_hash ^= Zobrist::zobrist_blacktomove;
  }
}

inline void Board::update_zobrist_ko(const int new_komove,
                                     const int old_komove) {
  m_hash ^= Zobrist::zobrist_ko[old_komove];
  m_hash ^= Zobrist::zobrist_ko[new_komove];
}

inline void Board::update_zobrist_pass(const int new_pass,
                                       const int old_pass) {
  m_hash ^= Zobrist::zobrist_pass[old_pass];
  m_hash ^= Zobrist::zobrist_pass[new_pass];
}

inline int Board::get_x(const int vtx) const {
  const int x = (vtx % m_letterboxsize) - 1;
  assert(x >= 0 && x < m_boardsize);
  return x;
}
inline int Board::get_y(const int vtx) const {
  const int y = (vtx / m_letterboxsize) - 1;
  assert(y >= 0 && y < m_boardsize);
  return y;
}
inline std::pair<int, int> Board::get_xy(const int vtx) const {
  const int x = get_x(vtx);
  const int y = get_y(vtx);
  return {x, y};
}


#endif
