#ifndef UCTNODE_H_INCLUDE
#define UCTNODE_H_INCLUDE

/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "Evaluation.h"
#include "GameState.h"

#include <atomic>
#include <cstdint>

class UCTNode;

#define POINTER_MASK (3ULL)

class Edge {
public:
  Edge(int vertex, float policy, float delta);
  ~Edge();
  Edge(Edge &&n);
  Edge(const Edge &) = delete;

  static size_t edge_tree_size;
  static size_t edge_node_count;
  static void increment_tree_size(size_t sz);
  static void decrement_tree_size(size_t sz);
  static void increment_tree_count(size_t ct);
  static void decrement_tree_count(size_t ct);

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
  bool is_invalid() const;
  bool is_uninflated() const;
  bool is_pointer(std::uint64_t) const;
  bool is_invalid(std::uint64_t) const;
  bool is_uninflated(std::uint64_t) const;

  UCTNode *get_node() const;

  UCTNode *operator->() const {
    assert(is_pointer());
    return read_ptr(m_pointer.load()); 
  }

  Edge &operator=(Edge &&n);

private:
  static constexpr std::uint64_t INVALID = 2;
  static constexpr std::uint64_t UNINFLATED = 1;
  static constexpr std::uint64_t POINTER = 0;

  float delta;
  float policy;
  int vertex;

  std::atomic<std::uint64_t> m_pointer{INVALID};

  UCTNode *read_ptr(uint64_t v) const;
};

inline bool Edge::is_pointer() const { is_pointer(m_pointer.load()); }

inline bool Edge::is_invalid() const { is_invalid(m_pointer.load()); }

inline bool Edge::is_uninflated() const { is_uninflated(m_pointer.load()); }

inline bool Edge::is_pointer(std::uint64_t v) const {
  return (v & POINTER_MASK) == POINTER;
}
inline bool Edge::is_invalid(std::uint64_t v) const {
  return (v & POINTER_MASK) == INVALID;
}
inline bool Edge::is_uninflated(std::uint64_t v) const {
  return (v & POINTER_MASK) == UNINFLATED;
}

inline int Edge::get_vertex() const { return vertex; }

inline float Edge::get_policy() const { return policy; }

inline UCTNode *Edge::read_ptr(uint64_t v) const {
  assert(is_pointer(v));
  return reinterpret_cast<UCTNode *>(v & ~(POINTER_MASK));
}

#define VIRTUAL_LOSS_COUNT (3)

class UCTNode {
public:
  static size_t node_tree_size;
  static size_t node_node_count;
  static void increment_tree_size(size_t sz);
  static void decrement_tree_size(size_t sz);
  static void increment_tree_count(size_t ct);
  static void decrement_tree_count(size_t ct);

  explicit UCTNode(float delta_loss, int vertex, float policy);
  UCTNode() = delete;
  ~UCTNode();

  bool expend_children(Evaluation &evaluation, GameState &state, float &eval,
                       float min_psa_ratio);

  void link_nodelist(std::vector<Network::PolicyVertexPair> &nodelist,
                     float min_psa_ratio);

  float get_raw_evaluation(int color) const;
  float get_accumulated_evals() const;
  float get_eval(int color) const;
  int get_visits() const;
  int get_vertex() const;
  float get_policy() const;

  void update(float eval);
  void accumulate_eval(float eval);
  void inflate_all_children();
  bool prune_child(int vtx);

  UCTNode *uct_select_child(int color, bool is_node);
  UCTNode *get_most_visits_child();

  void kill_superkos(const GameState &state);
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

  UCTNode *get_node();

private:
  enum Status : std::uint8_t { INVALID, PRUNED, ACTIVE };
  std::atomic<Status> m_status{ACTIVE};  

  int m_vertex;
  float m_policy;
  float m_raw_eval{0.0f};
  float m_delta_loss;

  std::atomic<float> m_squared_eval_diff{1e-4f};
  std::atomic<float> m_accumulated_evals{0.0};

  std::vector<Edge> m_children;

  std::atomic<int> m_virtual_loss{0};
  std::atomic<int> m_visits{0};

  enum class ExpandState : std::uint8_t { INITIAL = 0, EXPANDING, EXPANDED };

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

inline bool UCTNode::is_pruned() const { return m_status.load() == PRUNED; }

inline bool UCTNode::is_active() const { return m_status.load() == ACTIVE; }

inline bool UCTNode::is_valid() const { return m_status.load() != INVALID; }

#endif
