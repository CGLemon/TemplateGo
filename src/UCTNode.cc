#include "UCTNode.h"
#include "Board.h"
#include "Random.h"
#include "Utils.h"
#include "cfg.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <iterator>
#include <limits>
#include <mutex>
#include <numeric>
#include <utility>
#include <vector>


static std::mutex data_mutex;
static std::mutex node_tree_mutex;
static std::mutex edge_tree_mutex;


size_t DataBuffer::node_data_size = 0;

DataBuffer::DataBuffer() {
  filled_invalid_data();
  increment_size(sizeof(DataBuffer));
}

DataBuffer::~DataBuffer() {
  decrement_size(sizeof(DataBuffer));
}

void DataBuffer::filled_invalid_data() {
  vertex = Board::PASS;
  policy = 0.0f;
}


void DataBuffer::increment_size(size_t sz) {
  std::lock_guard<std::mutex> lock(data_mutex);
  node_data_size += sz;
}

void DataBuffer::decrement_size(size_t sz) {
  std::lock_guard<std::mutex> lock(data_mutex);
  node_data_size -= sz;
}

size_t Edge::edge_tree_size = 0;

Edge::Edge(std::shared_ptr<DataBuffer> data) {

  m_data = data;
  m_pointer.store(UNINFLATED);
  increment_tree_size(sizeof(Edge) +
                      sizeof(std::shared_ptr<Edge>));
}

Edge::~Edge() {
  const size_t scratch = sizeof(Edge) +
                         sizeof(std::shared_ptr<Edge>);

  if (is_pointer(m_pointer)) {
    increment_tree_size(scratch);
    delete read_ptr(m_pointer);
  }
  decrement_tree_size(scratch);
}

void Edge::set_policy(float policy) { 
  m_data->policy = policy;
}

void Edge::increment_tree_size(size_t sz) {
  std::lock_guard<std::mutex> lock(edge_tree_mutex);
  edge_tree_size += sz;
}

void Edge::decrement_tree_size(size_t sz) {
  std::lock_guard<std::mutex> lock(edge_tree_mutex);
  edge_tree_size -= sz;
}

bool Edge::acquire_inflating() {
  auto uninflated = UNINFLATED;
  auto newval = INFLATING;
  return m_pointer.compare_exchange_strong(uninflated, newval);
}

void Edge::inflate() {
  while (true) {
    auto v = m_pointer.load();
    if (is_pointer(v)) {
      return;
    }
    if (!acquire_inflating()) {
      continue;
    }
    auto new_ponter =
        reinterpret_cast<std::uint64_t>(new UCTNode(m_data)) |
        POINTER;
    auto old_ponter = m_pointer.exchange(new_ponter);
    decrement_tree_size(sizeof(Edge) + sizeof(std::shared_ptr<Edge>));
    assert(is_inflating(old_ponter));
  }
}

void Edge::prune_node() {
  inflate();
  auto v = m_pointer.load();
  auto n = read_ptr(v);
  n->set_active(false);
}

void Edge::kill_node() {
  inflate();
  auto v = m_pointer.load();
  auto n = read_ptr(v);
  n->invalinode();
}

bool Edge::is_pruned() const {
  auto v = m_pointer.load();
  if (!is_pointer(v)) {
    return false;
  }
  return read_ptr(v)->is_pruned();
}

bool Edge::is_active() const {
  auto v = m_pointer.load();
  if (!is_pointer(v)) {
    return true;
  }
  return read_ptr(v)->is_active();
}

bool Edge::is_valid() const {
  auto v = m_pointer.load();
  if (!is_pointer(v)) {
    return true;
  }
  return read_ptr(v)->is_valid();
}

int Edge::get_visits() const {
  auto v = m_pointer.load();
  if (!is_pointer(v)) {
    return 0;
  }
  return read_ptr(v)->get_visits();
}

float Edge::get_eval(int color, bool use_virtual_loss) const {
  auto v = m_pointer.load();
  if (!is_pointer(v)) {
    return 0.0f;
  }
  return read_ptr(v)->get_eval(color, use_virtual_loss);
}

UCTNode *Edge::get_node() const {
  auto v = m_pointer.load();
  assert(is_pointer(v));
  return read_ptr(v);
}

size_t UCTNode::node_tree_size = 0;

UCTNode::UCTNode(std::shared_ptr<DataBuffer> data) {

  m_data = data;

  increment_tree_size(sizeof(UCTNode));
}

UCTNode::~UCTNode() {
  decrement_tree_size(sizeof(UCTNode));
  assert(m_loading_threads.load() == 0);
}

bool UCTNode::expend_children(Evaluation &evaluation, GameState &state,
                              std::shared_ptr<NNOutputBuffer> &nn_output, float min_psa_ratio, bool kill_superkos) {


  // 防止無效的父節點產生，每個父節點至少要一個子節點
  assert(state.board.get_passes() < 2);
  /*
  if (state.board.get_passes() >= 2) {
    return false;
  }

  if (is_expending()) {
    return false;
  }
  */

  if (!acquire_expanding()) {
    return false;
  }
  const size_t boardsize = state.board.get_boardsize();
  const size_t intersections = boardsize * boardsize;

  const auto raw_netlist =
      evaluation.network_eval(state, Network::Ensemble::RANDOM_SYMMETRY);

  const float stm_eval = raw_netlist.winrate;
  const auto to_move = state.board.get_to_move();

  m_color = to_move;

  // 將網路輸出裝填進 buffer
  m_black_nn_output = std::make_shared<NNOutputBuffer>();
  if (to_move == Board::WHITE) {
    m_black_nn_output->eval = 1.0f - stm_eval;
  } else {
    m_black_nn_output->eval = stm_eval;
  }

  float max_score_prob = std::numeric_limits<float>::lowest();
  int max_final_score = std::numeric_limits<int>::lowest();
  for (auto idx = size_t{0}; idx < 2 * intersections; ++idx) {
    if (max_score_prob < raw_netlist.final_score[idx]) {
      max_score_prob = raw_netlist.final_score[idx];
      max_final_score = static_cast<int>(idx);
    }
  }

  max_final_score -= intersections;
  if (to_move == Board::WHITE) { 
    m_black_nn_output->final_score = 0 - max_final_score;
  } else {
    m_black_nn_output->final_score = max_final_score;
  }

  m_black_nn_output->ownership.reserve(intersections);
  m_black_nn_output->ownership.resize(intersections);

  for (auto idx = size_t{0}; idx < intersections; ++idx) {
    const float ownership = raw_netlist.ownership[idx];

    if (to_move == Board::WHITE) {
      m_black_nn_output->ownership[idx] = 0.0f - ownership;
    } else {
      m_black_nn_output->ownership[idx] = ownership;
    }
  }
  nn_output = m_black_nn_output;

  std::vector<Network::PolicyVertexPair> nodelist;

  float legal_accumulate = 0.0f;
  int legal_count = 0;
  for (auto i = size_t{0}; i < intersections; i++) {
    const int x = i % boardsize;
    const int y = i / boardsize;
    const auto vertex = state.board.get_vertex(x, y);
    const auto policy = raw_netlist.policy[i];
    if (state.board.is_legal(vertex, to_move)) {
      if (kill_superkos) {
	    auto smi_state = std::make_shared<GameState>(state);
	    if (!smi_state->play_move(vertex)) {
	      continue;
	    }
        if (smi_state->superko()) { 
          continue;
	    }
      }
      nodelist.emplace_back(policy, vertex);
      legal_accumulate += policy;
      legal_count++;
    }
  }
  assert(cfg_allow_pass_ratio != 0.0f);

  if (legal_count <= (intersections * cfg_allow_pass_ratio) || legal_accumulate <= 0.0f) {
    nodelist.emplace_back(raw_netlist.policy_pass, Board::PASS);
    legal_accumulate += raw_netlist.policy_pass;
  }

  assert(legal_accumulate != 0.0f);

  for (auto &node : nodelist) {
    node.first /= legal_accumulate;
  }

  link_nodelist(nodelist, min_psa_ratio);
  expand_done();

  return true;
}

void UCTNode::link_nodelist(std::vector<Network::PolicyVertexPair> &nodelist,
                            float min_psa_ratio) {
  std::stable_sort(rbegin(nodelist), rend(nodelist));

  const float min_psa = nodelist[0].first * min_psa_ratio;
  for (const auto &node : nodelist) {
    if (node.first < min_psa) {
      break;
    } else {
      auto data = std::make_shared<DataBuffer>();
      data->vertex = node.second;
      data->policy = node.first;

      m_children.emplace_back(std::move(std::make_shared<Edge>(data)));
    }
  }
  assert(!m_children.empty());
}

float UCTNode::get_raw_evaluation(int color) const {
  if (color == Board::BLACK) {
    return m_black_nn_output->eval;
  }
  return 1.0f - m_black_nn_output->eval;
}

int UCTNode::get_visits() const {
  return m_visits.load();
}

float UCTNode::get_accumulated_evals() const {
  return m_accumulated_black_evals.load();
}

int UCTNode::get_color() const {
 return m_color;
}

int UCTNode::get_vertex() const {
  return m_data->vertex;
}

float UCTNode::get_policy() const {
  return m_data->policy;
}

void UCTNode::increment_threads() {
  m_loading_threads++;
}

void UCTNode::decrement_threads() {
  m_loading_threads--;
  assert(m_loading_threads.load() >= 0);
}

int UCTNode::get_threads() const {
  return m_loading_threads.load();
}

int UCTNode::get_virtual_loss() const {
  const int threads = get_threads();
  const int virtual_loss = threads * threads * VIRTUAL_LOSS_COUNT;
 
  return virtual_loss;
}

float UCTNode::get_eval(int color, bool use_virtual_loss) const {
  int virtual_loss = get_virtual_loss();
  if (!use_virtual_loss) {
    virtual_loss = 0;
  }

  int visits = m_visits + virtual_loss;
  assert(visits >= 0);
  float accumulated_evals = get_accumulated_evals();
  if (color == Board::WHITE) {
     accumulated_evals += static_cast<float>(virtual_loss);
  }
  float eval = accumulated_evals / static_cast<float>(visits);
  if (color == Board::BLACK) {
    return eval;
  }
  return 1.0f - eval;
}

void UCTNode::inflate_all_children() {
  for (const auto &child : m_children) {
    child->inflate();
  }
}

bool UCTNode::prune_child(int vtx) {
  for (const auto &child : m_children) {
    if (child->get_vertex() == vtx) {
      child->prune_node();
      return true;
    }
  }
  return false;
}

UCTNode *UCTNode::uct_select_child(int color, bool is_root) {
  wait_expanded();
  assert(has_children());

  int parentvisits = 0;
  double total_visited_policy = 0.0f;
  for (const auto &child : m_children) {
    if (child->is_valid()) {
      parentvisits += child->get_visits();
      if (child->get_visits() > 0) {
        total_visited_policy += child->get_policy();
      }
    }
  }

  const double numerator =
      std::sqrt(double(parentvisits) *
                std::log(cfg_logpuct * double(parentvisits) + cfg_logconst));
  const double fpu_reduction =
      (is_root ? cfg_fpu_root_reduction : cfg_fpu_reduction) *
      std::sqrt(total_visited_policy);
  const double fpu_eval = double(get_raw_evaluation(color)) - fpu_reduction;

  std::shared_ptr<Edge> best_node = nullptr;
  //Edge *best_node = static_cast<Edge *>(nullptr);
  double best_value = std::numeric_limits<double>::lowest();

  for (const auto &child : m_children) {
    if (!child->is_active()) {
      continue;
    }

    double winrate = fpu_eval;

    if (child->is_pointer()) {
      if (child->get_node()->is_expending()) {
        winrate = -1.0f - fpu_reduction;
      } else if (child->get_visits() > 0) {
        winrate = child->get_eval(color);
      }
    }
    const double psa = child->get_policy();
    const double denom = 1.0 + child->get_visits();
    const double puct = cfg_puct * psa * (numerator / denom);
    const double value = winrate + puct;
    assert(value > std::numeric_limits<double>::lowest());

    if (value > best_value) {
      best_value = value;
      best_node = child;
    }
  }

  best_node->inflate();
  return best_node->get_node();
}

void UCTNode::dirichlet_noise(float epsilon, float alpha) {
  size_t child_cnt = m_children.size();

  auto dirichlet_buffer = std::vector<float>{};
  std::gamma_distribution<float> gamma(alpha, 1.0f);
  for (auto i = size_t{0}; i < child_cnt; i++) {
    float gen = gamma(Random<random_t::XoroShiro128Plus>::get_Rng());
    dirichlet_buffer.emplace_back(gen);
  }

  auto sample_sum =
      std::accumulate(std::begin(dirichlet_buffer), std::end(dirichlet_buffer), 0.0f);

  // If the noise vector sums to 0 or a denormal, then don't try to
  // normalize.
  if (sample_sum < std::numeric_limits<float>::min()) {
    return;
  }

  for (auto &v : dirichlet_buffer) {
    v /= sample_sum;
  }

  child_cnt = 0;
  for (const auto &child : m_children) {
    auto policy = child->get_policy();
    auto eta_a = dirichlet_buffer[child_cnt++];
    policy = policy * (1 - epsilon) + epsilon * eta_a;
    child->set_policy(policy);
  }
}

int UCTNode::get_most_visits_move() {
  wait_expanded();
  assert(has_children());

  int most_visits = std::numeric_limits<int>::lowest();
  int most_vertex = Board::NO_VERTEX;
  for (const auto &child : m_children) {
    const int node_visits = child->get_visits();
    if (node_visits > most_visits) {
      most_vertex = child->get_vertex();
      most_visits = node_visits;
    }
  }
  assert(most_vertex != Board::NO_VERTEX);

  return most_vertex;
}

UCTNode *UCTNode::get_most_visits_child() {
  wait_expanded();
  assert(has_children());

  int most_visits = std::numeric_limits<int>::lowest();
  std::shared_ptr<Edge> most_child = nullptr;
  //Edge *most_child = nullptr;
  for (const auto &child : m_children) { 
    const int node_visits = child->get_visits();
    if (node_visits > most_visits) {
      most_visits = node_visits;
      most_child = child;
    }
  }

  assert(most_child != nullptr);
  most_child->inflate();

  return most_child->get_node();
}

UCTNode *UCTNode::get_child(const int vtx) {
  wait_expanded();
  assert(has_children());

  std::shared_ptr<Edge> res = nullptr;
  //Edge *res = nullptr;

  for (const auto &child : m_children) { 
    const int vertex = child->get_vertex();
    if (vtx == vertex) {
      res = child;
      break;
    }
  }

  assert(res != nullptr);
  res->inflate();
  return res->get_node();
}

const std::vector<float>& UCTNode::get_ownership() const {
  return m_black_nn_output->ownership;
}

const std::vector<std::shared_ptr<Edge>>& UCTNode::get_children() const {
  return m_children;
}


void UCTNode::accumulate_eval(float eval) {
  Utils::atomic_add(m_accumulated_black_evals, eval);
}


void UCTNode::update(std::shared_ptr<NNOutputBuffer> nn_output) {

  const float eval = nn_output->eval;

  float old_eval = m_accumulated_black_evals.load();
  float old_visits = m_visits.load();
  float old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
  m_visits++;
  accumulate_eval(eval);
  float new_delta = eval - (old_eval + eval) / (old_visits + 1);
  // Welford's online algorithm for calculating variance.
  float delta = old_delta * new_delta;
  Utils::atomic_add(m_squared_eval_diff, delta);

  bool success = wait_update();
  if (success) {
    if (m_black_nn_output && m_black_nn_output != nn_output) {
      const size_t o_size = nn_output->ownership.size();
      for (auto idx = size_t{0}; idx < o_size; ++idx) {
        const auto owner = nn_output->ownership[idx];
        m_black_nn_output->ownership[idx] += owner;
      }
    }

    update_done();
  }
}

void UCTNode::prepare_root_node(Evaluation &evaluation, GameState &state, std::shared_ptr<NNOutputBuffer> &nn_output) {

  const bool kill_superkos = true;

  bool success = expend_children(evaluation, state, nn_output, 0.0f, kill_superkos);
  bool had_childen = has_children();
  assert(success && had_childen);

  if (success && had_childen) {
    inflate_all_children();
    size_t legal_move = m_children.size();
    if (cfg_dirichlet_noise) {
      float alpha = 0.03f * 361.0f / static_cast<float>(legal_move);
      dirichlet_noise(0.25f, alpha);
    }
  }
}

bool UCTNode::has_children() const { 
  return m_color != Board::EMPTY; 
}

void UCTNode::set_active(const bool active) {
  if (is_valid()) {
    m_status = active ? ACTIVE : PRUNED;
  }
}

void UCTNode::invalinode() {
  if (is_valid()) {
    m_status = INVALID;
  }
}

void UCTNode::increment_tree_size(size_t sz) {
  std::lock_guard<std::mutex> lock(node_tree_mutex);
  node_tree_size += sz;
}
void UCTNode::decrement_tree_size(size_t sz) {
  std::lock_guard<std::mutex> lock(node_tree_mutex);
  node_tree_size -= sz;
}

UCTNode *UCTNode::get_node() { 
  return this; 
}

float UCTNode::get_eval_variance(float default_var) const {
  return m_visits > 1 ? m_squared_eval_diff / (m_visits - 1) : default_var;
}

float UCTNode::get_eval_lcb(int color) const {
  // LCB github : https://github.com/leela-zero/leela-zero/issues/2411
  // Lower confidence bound of winrate.
  float visits = get_visits();
  if (visits < 2.0f) {
      // Return large negative value if not enough visits.
      return get_policy() - 1e6f;
  }
  float mean = get_eval(color, false);

  float stddev = std::sqrt(get_eval_variance(1.0f) / visits);
  float z = Utils::cached_t_quantile(visits - 1);

  return mean - z * stddev;
}


std::vector<std::pair<float, int>> UCTNode::get_lcb_list(const int color) {
  wait_expanded();
  assert(has_children());
  //assert(color == m_color);
  
  std::vector<std::pair<float, int>> list;
  inflate_all_children();

  for (const auto & child : m_children) {
    const auto visits = child->get_visits();
    const auto vertex = child->get_vertex();
    const auto lcb = child->get_node()->get_eval_lcb(color);
    if (visits > 0) {
      list.emplace_back(lcb, vertex);
    }
  }

  std::stable_sort(rbegin(list), rend(list));
  return list;
}

int UCTNode::get_best_move() {
  auto lcblist = get_lcb_list(m_color);

  float best_value = std::numeric_limits<float>::lowest();
  int best_move = Board::NO_VERTEX;

  for (auto &lcb : lcblist) {
    const auto lcb_value = lcb.first;
    const auto vertex = lcb.second;

    if (lcb_value > best_value) {
      best_value = lcb_value;
      best_move = vertex;
    }
  }

  if (lcblist.empty() && has_children()) {
     best_move = m_children[0] -> get_vertex();
  }

  assert(best_move != Board::NO_VERTEX);
  return best_move;
}

int UCTNode::randomize_first_proportionally(float random_temp) {

  int select_move = Board::NO_VERTEX;
  auto accum = double{0.0};
  auto accum_vector = std::vector<std::pair<double, int>>{};

  for (const auto& child : m_children) {
    const auto visits = child->get_visits();
    const auto vertex = child->get_vertex();

    if (visits > cfg_random_min_visits) {
      accum += std::pow((double)visits, (1.0 / random_temp));
      accum_vector.emplace_back(std::pair<double, int>(accum, vertex));
    }
  }

  auto distribution = std::uniform_real_distribution<double>{0.0, accum};
  auto pick = distribution(Random<random_t::XoroShiro128Plus>::get_Rng());
  auto size = accum_vector.size();

  for (auto idx = size_t{0}; idx < size; ++idx) {
    if (pick < accum_vector[idx].first) {
      select_move = accum_vector[idx].second;
      break;
    }
  }

  return select_move;
}

std::vector<std::pair<float, int>> UCTNode::get_winrate_list(const int color) {
  wait_expanded();
  assert(has_children());
  assert(color == m_color);

  inflate_all_children();
  std::vector<std::pair<float, int>> list;

  for (const auto & child : m_children) {
    const auto vertex = child->get_vertex();
    const auto visits = child->get_visits();
    const auto winrate = child->get_eval(color, false);
    if (visits > 0) {
      list.emplace_back(winrate, vertex);
    }
  }

  std::stable_sort(rbegin(list), rend(list));
  return list;
}

bool UCTNode::acquire_update() {
  auto expected = ExpandState::EXPANDED;
  auto newval = ExpandState::UPDATE;
  return m_expand_state.compare_exchange_strong(expected, newval);
}

bool UCTNode::wait_update() {
  while (!acquire_update()) {
    if (m_expand_state.load() == ExpandState::INITIAL) {
      return false;
    }
  }
  return true;
}

void UCTNode::update_done() {
  auto v = m_expand_state.exchange(ExpandState::EXPANDED);
  assert(v == ExpandState::UPDATE);
}

bool UCTNode::acquire_expanding() {
  auto expected = ExpandState::INITIAL;
  auto newval = ExpandState::EXPANDING;
  return m_expand_state.compare_exchange_strong(expected, newval);
}

void UCTNode::expand_done() {
  auto v = m_expand_state.exchange(ExpandState::EXPANDED);
  assert(v == ExpandState::EXPANDING);
}

void UCTNode::expand_cancel() {
  auto v = m_expand_state.exchange(ExpandState::INITIAL);
  assert(v == ExpandState::EXPANDING);
}

void UCTNode::wait_expanded() {
  while (true) {
    auto v = m_expand_state.load();
    if (v == ExpandState::EXPANDED || v == ExpandState::UPDATE) {
      break;
    }
  }
  //auto v = m_expand_state.load();
  //assert(v == ExpandState::EXPANDED);
}

size_t UCT_Information::get_memory_used() {
  return Edge::edge_tree_size + UCTNode::node_tree_size + DataBuffer::node_data_size;
}

void UCT_Information::dump_ownership(GameState &state, UCTNode *node) {
  const auto boardsize = state.board.get_boardsize();
  const auto color = state.board.get_to_move();
  const auto ownership = node->get_ownership();
  const auto visits = node-> get_visits() + 1;

  Utils::auto_printf("Ownership : \n");
  for (int y = 0; y < boardsize; ++y) {
    Utils::auto_printf(" ");
    for (int x = 0; x < boardsize; ++x) {
      const auto idx = state.board.get_index(x, y);
      const auto black_owner = (ownership[idx] / (float)visits + 1.0f) / 2.0f;
      auto owner = black_owner;
      if (color == Board::WHITE) {
        owner = 1.0f - owner;
      }
      Utils::auto_printf("%.4f ", owner);
    }
    Utils::auto_printf("\n");
  }
}


void UCT_Information::tree_stats() {
  const size_t edge_size = sizeof(Edge) + sizeof(std::shared_ptr<Edge>);
  const size_t node_size = sizeof(UCTNode);
  const size_t data_size = sizeof(DataBuffer);

  const size_t edge_count = Edge::edge_tree_size/edge_size;
  assert(Edge::edge_tree_size % edge_size == 0);

  const size_t node_count = UCTNode::node_tree_size/node_size;
  assert(UCTNode::node_tree_size % node_size == 0);

  //const size_t data_count = DataBuffer::node_data_size/data_size;
  assert(DataBuffer::node_data_size % data_size == 0);

  const size_t memory_size = get_memory_used();

  Utils::auto_printf("Nodes\n");
  Utils::auto_printf(" inflate nodes : %zu (counts)\n", node_count);
  Utils::auto_printf(" uninflate nodes : %zu (counts)\n", edge_count);
  Utils::auto_printf(" memory used : %.5f (MiB)\n", (float)memory_size / (1024 * 1024));
}

void UCT_Information::dump_stats(GameState &state, UCTNode *node, int cut_off) {
  const auto color = state.board.get_to_move();
  const auto lcblist = node->get_lcb_list(color);
  const auto parentsVisits = static_cast<float>(node->get_visits());
  assert(color == node->get_color());

  dump_ownership(state, node);

  Utils::auto_printf("Search List\n"); 
  int push = 0;
  for (auto &lcb : lcblist) {
    const auto lcb_value = lcb.first > 0.0f ? lcb.first : 0.0f;
    const auto vtx = lcb.second;
    
    auto child = node->get_child(vtx);
    const auto visits = child->get_visits();
    const auto pobability = child->get_policy();
    assert(visits != 0);
    
    const auto eval = child->get_eval(color, false);
    const auto move = state.vertex_to_string(vtx);
    const auto pv_string = move + " " + pv_to_srting(child, state);
    const float visit_ratio = static_cast<float>(visits) / parentsVisits;
    Utils::auto_printf("%4s -> %7d (V: %5.2f%%) (LCB: %5.2f%%) (N: %5.2f%%) (P: %5.2f%%) PV: %s\n", 
                        move.c_str(),
                        visits,
                        eval * 100.f, 
                        lcb_value * 100.f,
                        visit_ratio * 100.f,
                        pobability * 100.f,
                        pv_string.c_str());

    push++;
    if (push == cut_off) {
      Utils::auto_printf("     ...remain %d nodes\n", (int)lcblist.size() - cut_off);
      break;
    }
    
  }

  tree_stats();
}

std::string UCT_Information::pv_to_srting(UCTNode *node, GameState& state) {
  auto pvlist = std::vector<int>{};
  auto *next = node;
  while (next->has_children()) {
    int vtx = next->get_best_move();
    pvlist.emplace_back(vtx);
    next = next->get_child(vtx);
  }
  
  auto res = std::string{};
  for (auto &vtx : pvlist) {
    res += state.vertex_to_string(vtx);
    res += " ";
  }
  return res;
}

void UCT_Information::collect_nodes() {

}

bool Heuristic::pass_to_win(GameState &state) {
  if (state.board.get_last_move() == Board::PASS) {
    const auto color = state.board.get_to_move();
    const auto res = state.final_score();
    if (color == Board::BLACK && res > 0.f) {
      return true;
    } else if (color == Board::WHITE && res < 0.f) {
      return true;
    }
  }

  return false;
}

bool Heuristic::should_be_resign(GameState &state, UCTNode *node, float threshold) {
  const int bsize = state.board.get_boardsize();
  const int num_moves = state.board.get_movenum();
  const int color = state.board.get_to_move();
  assert(color == node->get_color());

  if (0.5f * (bsize * bsize) >  num_moves) {
    return false;
  }
  auto wlist = node->get_winrate_list(color);
  auto lcblist = node->get_lcb_list(color);

  const size_t size = wlist.size();
  assert(size == lcblist.size());

  for (auto i = size_t{0}; i < size; ++i) {
    const float winrate = wlist[i].first;
    const float lcb = lcblist[i].first;
    if (winrate > threshold || lcb > threshold) {
      return false;
    } 
  }

  return true;
}
