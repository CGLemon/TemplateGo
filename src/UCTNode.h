#ifndef UCTNODE_H_INCLUDE
#define UCTNODE_H_INCLUDE

#include "Evaluation.h"
#include "GameState.h"

#include <atomic>
#include <cstdint>
#include <memory>

struct DataBuffer {
  DataBuffer();
  ~DataBuffer();
  static size_t node_data_size;

  static void increment_size(size_t sz);
  static void decrement_size(size_t sz);

  float delta;
  float policy;
  int vertex;
  
  void filled_invalid_data();
};


class UCTNode;

#define POINTER_MASK (3ULL)

class Edge {
public:
  Edge(std::shared_ptr<DataBuffer> data);
  ~Edge();
  //Edge(Edge &&n);
  Edge(const Edge &) = delete;

  static size_t edge_tree_size;
  static void increment_tree_size(size_t sz);
  static void decrement_tree_size(size_t sz);

  void set_policy(float policy);

  int get_vertex() const;
  float get_policy() const;         
  int get_visits() const;
  float get_eval(int color) const; 

  void inflate();
  void kill_node();
  void prune_node();

  bool is_pruned() const;
  bool is_active() const;
  bool is_valid() const;

  bool is_pointer() const;
  bool is_inflating() const;
  bool is_uninflated() const;

  bool is_pointer(std::uint64_t) const;
  bool is_inflating(std::uint64_t) const;
  bool is_uninflated(std::uint64_t) const;

  bool acquire_inflating();
 
  UCTNode *get_node() const;

  //UCTNode *operator->() const {
  //  return read_ptr(m_pointer.load()); 
  //}

  //Edge &operator=(Edge &&n);

private:
  std::shared_ptr<DataBuffer> m_data;

  static constexpr std::uint64_t UNINFLATED = 2;
  static constexpr std::uint64_t INFLATING = 1;
  static constexpr std::uint64_t POINTER = 0;

  std::atomic<std::uint64_t> m_pointer{UNINFLATED};

  UCTNode *read_ptr(uint64_t v) const;
};

inline bool Edge::is_pointer() const { 
  return is_pointer(m_pointer.load());
}

inline bool Edge::is_inflating() const {
  return is_inflating(m_pointer.load());
}

inline bool Edge::is_uninflated() const {
  return is_uninflated(m_pointer.load());
}

inline bool Edge::is_pointer(std::uint64_t v) const {
  return (v & POINTER_MASK) == POINTER;
}

inline bool Edge::is_inflating(std::uint64_t v) const {
  return (v & POINTER_MASK) == INFLATING;
}

inline bool Edge::is_uninflated(std::uint64_t v) const {
  return (v & POINTER_MASK) == UNINFLATED;
}

inline int Edge::get_vertex() const { 
  return m_data->vertex; 
}

inline float Edge::get_policy() const { 
  return m_data->policy;
}

inline UCTNode *Edge::read_ptr(uint64_t v) const {
  assert(is_pointer(v));
  return reinterpret_cast<UCTNode *>(v & ~(POINTER_MASK));
}

#define VIRTUAL_LOSS_COUNT (3)

class UCTNode {
public:
  static size_t node_tree_size;
  static void increment_tree_size(size_t sz);
  static void decrement_tree_size(size_t sz);

  explicit UCTNode(std::shared_ptr<DataBuffer> data);
  UCTNode() = delete;
  ~UCTNode();

  bool expend_children(Evaluation &evaluation, GameState &state, float &eval,
                       float min_psa_ratio, bool kill_superkos = false);

  void link_nodelist(std::vector<Network::PolicyVertexPair> &nodelist,
                     float min_psa_ratio);

  float get_raw_evaluation(int color) const;
  float get_accumulated_evals() const;
  float get_eval(int color) const;
  int get_visits() const;
  int get_vertex() const;
  float get_policy() const;
  int get_color() const;

  void update(float eval);
  void accumulate_eval(float eval);
  void inflate_all_children();
  bool prune_child(int vtx);

  UCTNode *uct_select_child(int color, bool is_node);
  UCTNode *get_most_visits_child();
  UCTNode *get_node();

  // 一定要有對應 vtx 的子節點
  UCTNode *get_child(const int vtx);

  int get_most_visits_move();
  void dirichlet_noise(float epsilon, float alpha);
  float prepare_root_node(Evaluation &evaluation, GameState &state);

  void increment_virtual_loss();
  void decrement_virtual_loss();

  void set_active(const bool active);
  void invalinode();

  bool is_pruned() const;
  bool is_valid() const;
  bool is_active() const;
  bool has_children() const;

  bool is_expending() const;
  bool is_expended() const;
  bool expandable() const;

  float get_eval_lcb(int color) const;
  float get_eval_variance(float default_var) const;
  std::vector<std::pair<float, int>> get_lcb_list(const int color);
  int get_best_move();

  std::vector<std::pair<float, int>> get_winrate_list(const int color);

private:
  std::shared_ptr<DataBuffer> m_data;

  enum Status : std::uint8_t { 
      INVALID, 
      PRUNED,
      ACTIVE 
  };
  std::atomic<Status> m_status{ACTIVE};  

  int m_color{Board::EMPTY};

  std::atomic<int> m_visits{0}; // 節點訪問的次數
  //int m_vertex;
  //float m_policy;

  // m_raw_black_eval 黑方的網路輸出勝率
  float m_raw_black_eval{0.0f};
  float m_delta_loss;

  std::atomic<float> m_squared_eval_diff{1e-4f};

  // m_accumulated_black_evals 黑方經樹搜索累積的 Q 值
  std::atomic<float> m_accumulated_black_evals{0.0};

  std::vector<std::shared_ptr<Edge>> m_children;

  std::atomic<int> m_virtual_loss{0};

  enum class ExpandState : std::uint8_t { 
      INITIAL = 0, 
      EXPANDING, 
      EXPANDED 
  };

  std::atomic<ExpandState> m_expand_state{ExpandState::INITIAL};


  bool acquire_expanding();

  // EXPANDING -> DONE
  void expand_done();

  // EXPANDING -> INITIAL
  void expand_cancel();

  // wait until we are on EXPANDED state
  void wait_expanded();
};

inline bool UCTNode::expandable() const {
  return m_expand_state.load() == ExpandState::INITIAL;
}

inline bool UCTNode::is_expending() const {
  return m_expand_state.load() == ExpandState::EXPANDING;
}

inline bool UCTNode::is_expended() const {
  return m_expand_state.load() == ExpandState::EXPANDED;
}

inline bool UCTNode::is_pruned() const {
  return m_status.load() == PRUNED;
}

inline bool UCTNode::is_active() const {
  return m_status.load() == ACTIVE;
}

inline bool UCTNode::is_valid() const {
  return m_status.load() != INVALID;
}


class UCT_Information {
public:
  static size_t get_memory_used();
  
  static void tree_stats(); 

  static void dump_stats(GameState& state, UCTNode *node);

  static std::string pv_to_srting(UCTNode *node, GameState& state);

  static void collect_nodes();

};

class Heuristic {
public:
  static bool pass_to_win(GameState & state, float threshold);

  static bool should_be_resign(GameState & state, UCTNode *node, float threshold);

};

#endif
