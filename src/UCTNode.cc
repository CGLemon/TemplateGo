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


#include "UCTNode.h"
#include "Board.h"
#include "cfg.h"
#include "Utils.h"
#include "Random.h"

#include <cassert>
#include <cstdio>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <utility>
#include <vector>
#include <mutex>

static std::mutex tree_mutex;

size_t Edge::edge_tree_size = 0;
size_t Edge::edge_node_count = 0;

Edge::Edge(int put_vertex, float put_policy, float put_delta) {
	policy = put_policy;
	vertex = put_vertex;
	delta  = put_delta;
	m_pointer = UNINFLATED;

	increment_tree_size(sizeof(Edge));
	increment_tree_count(1);
}


Edge::Edge(Edge&& n) {
	auto nv = std::atomic_exchange(&n.m_pointer, INVALID);
    auto v = std::atomic_exchange(&m_pointer, nv);
	
	policy = n.policy;
	vertex = n.vertex;
	delta  = n.delta;
	

	increment_tree_size(sizeof(Edge));
	increment_tree_count(1);
    assert(v == INVALID);
}

Edge::~Edge() {
    if (is_pointer(m_pointer)) {
		increment_tree_size(sizeof(Edge));
		increment_tree_count(1);
        delete read_ptr(m_pointer);
    }
	decrement_tree_size(sizeof(Edge));
	decrement_tree_count(1);
}

void Edge::set_policy(float put_policy) {
	policy = put_policy;
}



void Edge::increment_tree_size(size_t sz) {
	std::lock_guard<std::mutex> lock(tree_mutex);
	edge_tree_size += sz;
}

void Edge::decrement_tree_size(size_t sz) {
	std::lock_guard<std::mutex> lock(tree_mutex);
	edge_tree_size -= sz;
}

void Edge::increment_tree_count(size_t ct) {
	std::lock_guard<std::mutex> lock(tree_mutex);
	edge_node_count += ct;
}

void Edge::decrement_tree_count(size_t ct) {
	std::lock_guard<std::mutex> lock(tree_mutex);
	edge_node_count -= ct;
}


void Edge::inflate() {
	
	while(true) {
		if (!is_uninflated(m_pointer.load())) return;
		auto ori_ponter = m_pointer.load();
		auto new_ponter = reinterpret_cast<std::uint64_t>(
            new UCTNode(delta, vertex, policy)
        ) | POINTER;
		bool success = m_pointer.compare_exchange_strong(ori_ponter, new_ponter);
		if (success) {
			decrement_tree_size(sizeof(Edge));
			decrement_tree_count(1);
            return;
        } else {
            delete read_ptr(new_ponter);
        }
	}
}

Edge& Edge::operator=(Edge&& n) {
	auto nv = std::atomic_exchange(&n.m_pointer, INVALID);
	auto v = std::atomic_exchange(&m_pointer, nv);
	if (is_pointer(v)) {
	    delete read_ptr(v);
	}
	return *this;
}


void Edge::prune_node() {
	inflate();
	auto n = read_ptr(m_pointer.load());
	n->set_active(false);
}

void Edge::kill_node() {
	inflate();
	auto n = read_ptr(m_pointer.load());
	n->invalinode();
}

bool Edge::is_pruned() const {
	auto v = m_pointer.load();
	if (!is_pointer(v)) {return false;}
	return read_ptr(v)->is_pruned();
}

bool Edge::is_active() const {
	auto v = m_pointer.load();
	if (!is_pointer(v)) {return true;}
	return read_ptr(v)->is_active();
}

bool Edge::is_valid() const {
	auto v = m_pointer.load();
	if (!is_pointer(v)) {return true;}
	return read_ptr(v)->is_valid();
}

int Edge::get_visits() const {
	auto v = m_pointer.load();
	if (!is_pointer(v)) {return 0;}
	return read_ptr(v)->get_visits();
}

float Edge::get_eval(int color) const {
	auto v = m_pointer.load();
	assert(is_pointer(v));
	return read_ptr(v)->get_eval(color);
}

UCTNode* Edge::get_node() const {
	auto v = m_pointer.load();
	assert(is_pointer(v));
	return read_ptr(v);
}



size_t UCTNode::node_tree_size = 0;
size_t UCTNode::node_node_count = 0;


UCTNode::UCTNode(float delta_loss, int vertex, float policy) {
	m_delta_loss = delta_loss;
	m_vertex = vertex;
	m_policy = policy;
	increment_tree_size(sizeof(UCTNode));
	increment_tree_count(1);
}

UCTNode::~UCTNode() {
	decrement_tree_size(sizeof(UCTNode));
    decrement_tree_count(1);
}

bool UCTNode::expend_children(Evaluation & evaluation, GameState& state,
						 float & eval, float min_psa_ratio) {

	if (!expandable()) {
		return false;
	}

	if (state.board.get_passes() >= 2) {
        return false;
    }

	// acquire the lock
    if (!acquire_expanding()) {
        return false;
    }

	const auto raw_netlist = evaluation.network_eval(
        state, Network::Ensemble::RANDOM_SYMMETRY);

	const float stm_eval = raw_netlist.winrate[0];
    const auto to_move = state.board.get_to_move();
	if (to_move == Board::WHITE) {
        m_raw_eval = 1.0f - stm_eval;
    } else {
        m_raw_eval = stm_eval;
    }
    eval = m_raw_eval;
	
	std::vector<Network::PolicyVertexPair> nodelist;
	
	float legal_sum = 0.0f;
	int legal_count = 0;
    for (int i = 0; i < NUM_INTERSECTIONS; i++) {
        const int x = i % BOARD_SIZE;
        const int y = i / BOARD_SIZE;
        const int vertex = state.board.get_vertex(x, y);
        if (state.board.is_legal(vertex, to_move)) {
            nodelist.emplace_back(raw_netlist.policy[i], vertex);
            legal_sum += raw_netlist.policy[i];
			legal_count++;
        }
    }
	assert(cfg_allow_pass_ratio != 0.0f);
	if (legal_count <= (NUM_INTERSECTIONS* cfg_allow_pass_ratio)) {
		nodelist.emplace_back(raw_netlist.policy_pass, Board::PASS);
		legal_sum += raw_netlist.policy_pass;
	}
	for (auto& node : nodelist) {
    	node.first /= legal_sum;
	}
	
	link_nodelist(nodelist, min_psa_ratio);
	expand_done();
	return true;
}

void UCTNode::link_nodelist(std::vector<Network::PolicyVertexPair>& nodelist, float min_psa_ratio) {
	
	if (nodelist.empty()) {
        return;
    }
	std::stable_sort(rbegin(nodelist), rend(nodelist));
	//const float max_psa = nodelist[0].first * min_psa_ratio;
	const float min_psa = nodelist[0].first * min_psa_ratio;
    for (const auto& node : nodelist) {
        if (node.first < min_psa) {
            break;
        } else {
            m_children.emplace_back(node.second, node.first, m_delta_loss* cfg_delta_attenuation_ratio);
        }
    }
}

int UCTNode::get_visits() const {
	return m_visits;
}

float UCTNode::get_raw_evaluation(int color) const {
	if (color == Board::BLACK) {
		return m_raw_eval;
	}
	return 1.0f - m_raw_eval;
}

float UCTNode::get_accumulated_evals() const {
	return  m_accumulated_evals;
}

int UCTNode::get_vertex() const {
	return m_vertex;
}
float UCTNode::get_policy() const {
	return m_policy;
}

void UCTNode::increment_virtual_loss() {
	m_virtual_loss += VIRTUAL_LOSS_COUNT;
}

void UCTNode::decrement_virtual_loss() {
	m_virtual_loss -= VIRTUAL_LOSS_COUNT;
}


float UCTNode::get_eval(int color) const {
	int visits = m_visits + m_virtual_loss;
	assert(visits > 0);
	float accumulated_evals = get_accumulated_evals();
	if (color == Board::WHITE) {
        //accumulated_evals += static_cast<float>(m_virtual_loss);
    }
	float eval = accumulated_evals / static_cast<float>(visits);
	if (color == Board::BLACK) {
		return eval;
	}
	return 1.0f - eval;
}

void UCTNode::inflate_all_children() {
	for (auto &childe : m_children) {
		childe.inflate();
	}
}

bool UCTNode::prune_child(int vtx) {
	for (auto &childe : m_children) {
		if (childe.get_vertex() == vtx) {
			childe.prune_node();
			return true;		
		}
	}
	return false;
}



UCTNode* UCTNode::uct_select_child(int color, bool is_root) {
	wait_expanded();

	
	int parentvisits = 0;
	double total_visited_policy = 0.0f;
	for (auto &child : m_children) {
		if (child.is_valid()) {
			parentvisits += child.get_visits();
			if (child.get_visits() > 0) {
                total_visited_policy += child.get_policy();
            }
		}
	}

	const double numerator = std::sqrt(double(parentvisits) *
            std::log(cfg_logpuct * double(parentvisits) + cfg_logconst));
	const double fpu_reduction = (is_root ? cfg_fpu_root_reduction : cfg_fpu_reduction)
								 * std::sqrt(total_visited_policy);
	const double fpu_eval = double(get_raw_evaluation(color)) - fpu_reduction;

	Edge* best_node = static_cast<Edge*>(nullptr);
	double best_value = std::numeric_limits<double>::lowest();

	for (auto &child : m_children) {
		if (!child.is_active()) {
			continue;
		}

		double winrate = fpu_eval;

		if (child.is_pointer() && child->m_expand_state.load() == ExpandState::EXPANDING) {
            winrate = -1.0f - fpu_reduction;
        } else if (child.get_visits() > 0) {
            winrate = child.get_eval(color);
        }
		const auto psa = child.get_policy();
        const auto denom = 1.0 + child.get_visits();
        const auto puct = cfg_puct * psa * (numerator / denom);
        const auto value = winrate + puct;
        assert(value > std::numeric_limits<double>::lowest());

        if (value > best_value) {
            best_value = value;
            best_node = &child;
        }
	}

	best_node->inflate();
	return best_node->get_node();
}

void UCTNode::dirichlet_noise(float epsilon, float alpha) {
    int child_cnt = m_children.size();

    auto dirichlet_vector = std::vector<float>{};
    std::gamma_distribution<float> gamma(alpha, 1.0f);
    for (size_t i = 0; i < child_cnt; i++) {
        dirichlet_vector.emplace_back(gamma(Random::get_Rng()));
    }

    auto sample_sum = std::accumulate(begin(dirichlet_vector),
                                      end(dirichlet_vector), 0.0f);

    // If the noise vector sums to 0 or a denormal, then don't try to
    // normalize.
    if (sample_sum < std::numeric_limits<float>::min()) {
        return;
    }

    for (auto& v : dirichlet_vector) {
        v /= sample_sum;
    }

    child_cnt = 0;
    for (auto& child : m_children) {
        auto policy = child.get_policy();
        auto eta_a = dirichlet_vector[child_cnt++];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
        child.set_policy(policy);
    }
}


void UCTNode::kill_superkos(const GameState& state) {
    Edge *pass_child = nullptr;
    size_t valid_count = 0;

    for (auto& child : m_children) {
        auto vtx = child.get_vertex();
        if (vtx != Board::PASS) {
            auto smi_state = std::make_shared<GameState>(state);
            if (!smi_state->play_move(vtx)) {
				child.kill_node();
			}

            if (smi_state->superko()) {
                child.kill_node();
            }
        } else {
            pass_child = &child;
        }
        if (child.is_valid()) {
            valid_count++;
        }
    }
	/*
    if (valid_count > 1 && pass_child &&
            !state.is_move_legal(state.get_to_move(), Board::PASS)) {
        // Remove the PASS node according to "avoid" -- but only if there are
        // other valid nodes left.
        (*pass_child)->invalidate();
    }
	*/
    // Now do the actual deletion.
    m_children.erase(
        std::remove_if(begin(m_children), end(m_children),
                       [](const auto &child) { return !child.is_valid();}),
        end(m_children)
    );
}

int UCTNode::get_most_visits_move() {
    wait_expanded();

    assert(!m_children.empty());

    int most_visits = std::numeric_limits<int>::lowest();
	int most_vertex = Board::NO_VERTEX;
	for (auto& child : m_children) {
		const int node_visits = child.get_visits();
		if (node_visits > most_visits) {
			most_vertex = child.get_vertex();
			most_visits = node_visits;
		}
	}
	assert(most_vertex != Board::NO_VERTEX);

    return most_vertex;
}

void UCTNode::accumulate_eval(float eval) {
	Utils::atomic_add(m_accumulated_evals, eval);
}

void UCTNode::update(float eval) {
	float old_eval = m_accumulated_evals;
    float old_visits = m_visits.load();
    float old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
    m_visits++;
    accumulate_eval(eval);
    float new_delta = eval - (old_eval + eval) / (old_visits + 1);
    // Welford's online algorithm for calculating variance.
    float delta = old_delta * new_delta;
    Utils::atomic_add(m_squared_eval_diff, delta);
}

float UCTNode::prepare_root_node(Evaluation & evaluation, GameState & state) {
	
	float root_eval;
	bool success = expend_children(evaluation, state, root_eval, 0.0f);
	bool had_childen = has_children();
	assert(success && had_childen);

	if (success && had_childen) {
		inflate_all_children();
		kill_superkos(state);

		if (cfg_noise) {
			float alpha = 0.03f * 361.0f / NUM_INTERSECTIONS;
			dirichlet_noise(0.25f, alpha);
		}
	}

	return root_eval;
}

bool UCTNode::has_children() const {
	return m_children.size();
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
	std::lock_guard<std::mutex> lock(tree_mutex);
	node_tree_size += sz;
}
void UCTNode::decrement_tree_size(size_t sz) {
	std::lock_guard<std::mutex> lock(tree_mutex);
	node_tree_size -= sz;
}

void UCTNode::increment_tree_count(size_t ct) {
	std::lock_guard<std::mutex> lock(tree_mutex);
	node_node_count += ct;
}
void UCTNode::decrement_tree_count(size_t ct)  {
	std::lock_guard<std::mutex> lock(tree_mutex);
	node_node_count -= ct;
}

bool UCTNode::is_expended() const {
	return ExpandState::EXPANDED == m_expand_state;
}

bool UCTNode::expandable() const {
	return ExpandState::INITIAL == m_expand_state;
}

UCTNode* UCTNode::get_node() {
	return this;
}

bool UCTNode::acquire_expanding() {
    auto expected = ExpandState::INITIAL;
    auto newval = ExpandState::EXPANDING;
    return m_expand_state.compare_exchange_strong(expected, newval);
}

void UCTNode::expand_done() {
    auto v = m_expand_state.exchange(ExpandState::EXPANDED);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::expand_cancel() {
    auto v = m_expand_state.exchange(ExpandState::INITIAL);
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDING);
}
void UCTNode::wait_expanded() {
    while (m_expand_state.load() == ExpandState::EXPANDING) {}
    auto v = m_expand_state.load();
#ifdef NDEBUG
    (void)v;
#endif
    assert(v == ExpandState::EXPANDED);
}
