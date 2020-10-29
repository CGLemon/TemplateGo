#include "UCTNode.h"
#include "Board.h"
#include "Random.h"
#include "Utils.h"
#include "config.h"

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

UCTNode::UCTNode(std::shared_ptr<UCTData> data) {
    m_data = data;
    assert(m_data->parameters);
    m_accumulated_black_ownership.fill(0.0f);
    m_raw_black_ownership.fill(0.0f);
}

UCTNode::~UCTNode() {
    assert(m_loading_threads.load() == 0);
}

std::shared_ptr<SearchParameters> UCTNode::parameters() const {
    return m_data->parameters;
}

bool UCTNode::expend_children(Evaluation &evaluation,
                              GameState &state,
                              std::shared_ptr<NNOutput> &nn_output,
                              const float min_psa_ratio, const bool is_root) {

    assert(state.get_passes() < 2);

    if (!acquire_expanding()) {
        return false;
    }

    const size_t boardsize = (size_t)state.get_boardsize();
    const size_t intersections = (size_t)state.get_intersections();

    auto raw_netlist =
        evaluation.network_eval(state, Network::Ensemble::RANDOM_SYMMETRY);

    m_color = state.get_to_move();
    link_nn_output(state, raw_netlist, nn_output, m_color);


    auto nodelist = std::vector<Network::PolicyVertexPair>{};
    bool allow_pass = true;
    float legal_accumulate = 0.0f;
    size_t legal_count = 0;

    for (auto i = size_t{0}; i < intersections; i++) {
        const auto x = i % boardsize;
        const auto y = i / boardsize;
        const auto vertex = state.get_vertex(x, y);
        const auto policy = raw_netlist.policy[i];


        // We block some move at root.
        if (state.is_legal(vertex, m_color, is_root == true ?
                 Board::avoid_t::SEARCH_BLOCK : Board::avoid_t::NONE)) {
            if (is_root) {
	            auto smi_state = std::make_shared<GameState>(state);
                bool take_move = smi_state->board.is_take_move(vertex, m_color);
                bool eye_move = smi_state->board.is_eye(vertex, m_color);
                
	            if (!smi_state->play_move(vertex)) {
	                continue;
	            }

                // If the node is superko, cutting it.
                if (smi_state->superko()) { 
                    continue;
	            }

                if (take_move) {
                    // If the move can take stone(s), cutting the pass node.
                    allow_pass = false;
                } else if (eye_move) {
                     // If the node is unnecessary suicide(over ten stones), cutting it.
                     if (smi_state->board.get_libs(vertex) == 1 && smi_state->board.get_stones(vertex) >= 10) {
                         continue;
                     }
                }
            }
            nodelist.emplace_back(policy, vertex);
            legal_accumulate += policy;
            legal_count++;
        }
    }
    assert(parameters()->allowed_pass_ratio != 0.0f);

    if (is_root && allow_pass && parameters()->collect) {
        // Avoid negative influence on training data,
        // if there are seki point on the board, cutting the pass node.
        auto alive_seki = state.board.get_alive_seki(m_color);
        allow_pass = alive_seki.empty();
    }

    if (legal_accumulate <= 0.0f) {
        allow_pass = true;
    }

    if (allow_pass && legal_count <= (int)(intersections * parameters()->allowed_pass_ratio)) {
        if (Heuristic::pass_to_win(state)) {
            // If pass node is win, cutting all children except pass node.
            nodelist.clear();
            legal_accumulate = 0.0f;
            legal_count = 0;
        }

        nodelist.emplace_back(raw_netlist.policy_pass, Board::PASS);
        legal_accumulate += raw_netlist.policy_pass;

        if (legal_accumulate <= 0.0f && !nodelist.empty()) {
            // Avoid pass policy is zero.
            legal_accumulate = 1.0f;
        }
    }

    assert(legal_accumulate != 0.0f);

    for (auto &node : nodelist) {
        node.first /= legal_accumulate;
    }

    link_nodelist(nodelist, min_psa_ratio);
    expand_done();

    return true;
}

void UCTNode::link_nodelist(std::vector<Network::PolicyVertexPair> &nodelist, float min_psa_ratio) {

    std::stable_sort(rbegin(nodelist), rend(nodelist));

    const float min_psa = nodelist[0].first * min_psa_ratio;
    for (const auto &node : nodelist) {
        if (node.first < min_psa) {
            break;
        } else {
            auto data = std::make_shared<UCTData>();
            data->vertex = node.second;
            data->policy = node.first;
            data->parameters = parameters();
            m_children.emplace_back(std::move(std::make_shared<UCTNodePointer>(data)));
        }
    }
    assert(!m_children.empty());
}

void UCTNode::link_nn_output(GameState &state,
                             const Evaluation::NNeval &raw_netlist,
                             std::shared_ptr<NNOutput> &nn_output, const int color){

    const size_t intersections = state.get_intersections();

    // Put network evaluation into node.
    const auto winrate = Model::get_winrate(state, raw_netlist);
    const auto stm_eval = winrate;
    if (color == Board::WHITE) {
        m_raw_black_eval = 1.0f - stm_eval;
    } else {
        m_raw_black_eval = stm_eval;
    }

    // ownership
    m_raw_black_ownership.fill(0.0f);
    for (auto idx = size_t{0}; idx < intersections; ++idx) {
        const float ownership = raw_netlist.ownership[idx];
        if (color == Board::WHITE) {
            m_raw_black_ownership[idx] = 0.0f - ownership;
        } else {
            m_raw_black_ownership[idx] = ownership;
        }
    }

    // final score
    if (color == Board::WHITE) {
        m_raw_black_final_score = 0.0f - raw_netlist.final_score;
    } else {
        m_raw_black_final_score = raw_netlist.final_score;
    }
    if (!nn_output) {
        nn_output=std::make_shared<NNOutput>();
    }
    std::copy(m_raw_black_ownership.begin(), m_raw_black_ownership.end(), nn_output->ownership.begin());
    nn_output->final_score = m_raw_black_final_score;
    nn_output->eval = m_raw_black_eval;
}

void UCTNode::from_nn_output(std::shared_ptr<NNOutput> nn_output) {
    if (nn_output && !m_terminal.load()) {
        // This node is double pass node.
        m_terminal.store(true);
        std::copy(nn_output->ownership.begin(), nn_output->ownership.end(), m_raw_black_ownership.begin());
        m_raw_black_final_score = nn_output->final_score;
        m_raw_black_eval = nn_output->eval;
    }
}

float UCTNode::get_raw_evaluation(const int color) const {
    if (color == Board::BLACK) {
        return m_raw_black_eval;
    }

    return 1.0f - m_raw_black_eval;
}

int UCTNode::get_vertex() const {
    return m_data->vertex;
}

float UCTNode::get_policy() const {
    return m_data->policy;
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
    const auto threads = get_threads();
    const auto virtual_loss = threads * VIRTUAL_LOSS_COUNT;
    return virtual_loss;
}

float UCTNode::get_eval(const int color, bool use_virtual_loss) const {

    auto virtual_loss = get_virtual_loss();
    auto visits = get_visits();

    if (use_virtual_loss) {
        // If this node is seaching, punish this node.
        visits += virtual_loss;
    }

    assert(visits >= 0);
    auto accumulated_evals = get_accumulated_evals();
    if (color == Board::WHITE && use_virtual_loss) {
        accumulated_evals += static_cast<float>(virtual_loss);
    }
    auto eval = accumulated_evals / static_cast<float>(visits);
    if (color == Board::BLACK) {
        return eval;
    }
    return 1.0f - eval;
}

float UCTNode::get_final_score(const int color) const {
    const auto visits = get_visits();
    if (visits == 0) {
        return 0.0f;
    }

    const auto final_score = m_accumulated_black_finalscore.load() / visits;
    if (color == Board::BLACK) {
        return final_score;
    }
    return 0.0f - final_score;
}

int UCTNode::get_most_visits_move() {
    wait_expanded();
    assert(has_children());

    int most_visits = std::numeric_limits<int>::lowest();
    int most_vertex = Board::NO_VERTEX;
    for (const auto &child : m_children) {
        if (!child->get()) {
            continue;
        }

        const int node_visits = child->get()->get_visits();
        if (node_visits > most_visits) {
            most_vertex = child->get()->get_vertex();
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
    std::shared_ptr<UCTNodePointer> most_child = nullptr;

    for (const auto &child : m_children) {
        if (!child->get()) {
            continue;
        }
        const int node_visits = child->get()->get_visits();
        if (node_visits > most_visits) {
            most_visits = node_visits;
            most_child = child;
        }
    }

    assert(most_child != nullptr);
    most_child->inflate();

    return most_child->get();
}

UCTNode *UCTNode::get_child(const int vtx) {
    wait_expanded();
    assert(has_children());

    std::shared_ptr<UCTNodePointer> res = nullptr;

    for (const auto &child : m_children) { 
        const int vertex = child->data()->vertex;
        if (vtx == vertex) {
            res = child;
            break;
        }
    }

    assert(res != nullptr);
    res->inflate();
    return res->get();
}

int UCTNode::randomize_first_proportionally(float random_temp) {

    int select_move = Board::NO_VERTEX;
    auto accum = double{0.0};
    auto accum_vector = std::vector<std::pair<double, int>>{};

    for (const auto &child : m_children) {
        const auto visits = child->get()->get_visits();
        const auto vertex = child->get()->get_vertex();
        if (visits > parameters()->random_min_visits) {
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

std::array<float, NUM_INTERSECTIONS> UCTNode::get_ownership(const int color) const {
    auto visits = get_visits();
    auto ownership = m_accumulated_black_ownership;
    for (auto &owner : ownership) {
        owner /= visits;
    }

    if (color == Board::WHITE) {
        for (auto &owner : ownership) {
            owner = 0.0f - owner;
        }
    }

    return ownership;
}

const std::vector<std::shared_ptr<UCTNode::UCTNodePointer>>& UCTNode::get_children() const {
    return m_children;
}

float UCTNode::get_eval_variance(float default_var) const {
    return m_visits > 1 ? m_squared_eval_diff / (m_visits - 1) : default_var;
}

float UCTNode::get_eval_lcb(const int color) const {
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
        const auto visits = child->get()->get_visits();
        const auto vertex = child->get()->get_vertex();
        const auto lcb = child->get()->get_eval_lcb(color);
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
        best_move = m_children[0]->get()->get_vertex();
    }

    assert(best_move != Board::NO_VERTEX);
    return best_move;
}


std::vector<std::pair<float, int>> UCTNode::get_winrate_list(const int color) {
    wait_expanded();
    assert(has_children());
    assert(color == m_color);

    inflate_all_children();
    std::vector<std::pair<float, int>> list;

    for (const auto &child : m_children) {
        const auto vertex = child->get()->get_vertex();
        const auto visits = child->get()->get_visits();
        const auto winrate = child->get()->get_eval(color, false);
        if (visits > 0) {
            list.emplace_back(winrate, vertex);
        }
    }

    std::stable_sort(rbegin(list), rend(list));
    return list;
}

UCTNode *UCTNode::get() {
    return this;
}

void UCTNode::set_policy(const float policy) {
    m_data->policy = policy;
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
}

bool UCTNode::has_children() const { 
    return m_color != Board::INVAL; 
}

bool UCTNode::expandable() const {
    return m_expand_state.load() == ExpandState::INITIAL;
}

bool UCTNode::is_expending() const {
    return m_expand_state.load() == ExpandState::EXPANDING;
}

bool UCTNode::is_expended() const {
    return m_expand_state.load() == ExpandState::EXPANDED;
}

bool UCTNode::is_pruned() const {
    return m_status.load() == PRUNED;
}

bool UCTNode::is_active() const {
    return m_status.load() == ACTIVE;
}

bool UCTNode::is_valid() const {
    return m_status.load() != INVALID;
}

void UCTNode::inflate_all_children() {
    for (const auto &child : m_children) {
        child->inflate();
    }
}

void UCTNode::set_active(const bool active) {
    if (is_valid()) {
        m_status = active ? ACTIVE : PRUNED;
    }
}

// If the node is illegal(eg. superko), we can set it as invalid node.
void UCTNode::invalinode() {
    if (is_valid()) {
        m_status = INVALID;
    }
}

bool UCTNode::prune_child(const int vtx) {
    for (const auto &child : m_children) {
        if (child->data()->vertex == vtx) {
            child->inflate();
            child->get()->set_active(false);
            return true;
        }
    }
    return false;
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
    // Be Sure all node are expended.
    inflate_all_children();
    for (const auto &child : m_children) {
        auto policy = child->get()->get_policy();
        auto eta_a = dirichlet_buffer[child_cnt++];
        policy = policy * (1 - epsilon) + epsilon * eta_a;
        child->get()->set_policy(policy);
    }
}

void UCTNode::prepare_root_node(Evaluation &evaluation,
                                GameState &state,
                                std::shared_ptr<NNOutput> &nn_output) {

    const bool is_root = true;
    bool success = expend_children(evaluation, state, nn_output, 0.0f, is_root);

    bool had_childen = has_children();
    assert(success && had_childen);

    if (success && had_childen) {
        inflate_all_children();
        size_t legal_move = m_children.size();
        if (parameters()->dirichlet_noise) {
            float alpha = 0.03f * 361.0f / static_cast<float>(legal_move);
            dirichlet_noise(0.25f, alpha);
        }
    }
}

UCTNode *UCTNode::uct_select_child(const int color, bool is_root) {
    wait_expanded();
    assert(has_children());

    int parentvisits = 0;
    double total_visited_policy = 0.0f;
    for (const auto &child : m_children) {
        if (!child->get()) {
            continue;
        }    
        if (child->get()->is_valid()) {
            parentvisits += child->get()->get_visits();
            if (child->get()->get_visits() > 0) {
                total_visited_policy += child->get()->get_policy();
            }
        }
    }

    const auto fpu_root_reduction  = parameters()->fpu_root_reduction;
    const auto fpu_child_reduction = parameters()->fpu_reduction;
    const auto logpuct             = parameters()->logpuct;
    const auto logconst            = parameters()->logconst;
    const auto cpuct               = parameters()->puct;

    const double numerator =
        std::sqrt(double(parentvisits) *
            std::log(logpuct * double(parentvisits) + logconst));
    const double fpu_reduction =
        (is_root ? fpu_root_reduction : fpu_child_reduction) *
        std::sqrt(total_visited_policy);
    const double fpu_eval = double(get_raw_evaluation(color)) - fpu_reduction;

    const double mean_score = get_mean_score(color);

    std::shared_ptr<UCTNodePointer> best_node = nullptr;
    double best_value = std::numeric_limits<double>::lowest();

    for (const auto &child : m_children) {
        // Check the node is pointer or not.
        // If not, we can not get most data from child.
        bool is_pointer = child->get();

        // If the node was pruned. Skip this time,
        if (is_pointer && !child->get()->is_active()) {
            continue;
        }

        double winrate = fpu_eval + get_score_utility(color, mean_score);
        if (is_pointer) {
            if (child->get()->is_expending()) {
                winrate = -1.0f - fpu_reduction;
            } else if (child->get()->get_visits() > 0) {
                winrate = child->get()->get_eval(color) +
                              child->get()->get_score_utility(color, mean_score);
            }
        }   
        double denom = 1.0;
        if (is_pointer) {
            denom += child->get()->get_visits();
        }

        const double psa = child->data()->policy;
        const double puct = cpuct * psa * (numerator / denom);
        const double value = winrate + puct;
        assert(value > std::numeric_limits<double>::lowest());

        if (value > best_value) {
            best_value = value;
            best_node = child;
        }
    }

    best_node->inflate();
    return best_node->get();
}

void UCTNode::accumulate_eval(float eval) {
    Utils::atomic_add(m_accumulated_black_evals, eval);
}

void UCTNode::update(std::shared_ptr<NNOutput> nn_output) {

    const float eval = nn_output->eval;

    float old_eval = m_accumulated_black_evals.load();
    float old_visits = m_visits.load();
    float old_delta = old_visits > 0 ? eval - old_eval / old_visits : 0.0f;
    m_visits++;
    Utils::atomic_add(m_accumulated_black_evals, eval);

    float new_delta = eval - (old_eval + eval) / (old_visits + 1);
    // Welford's online algorithm for calculating variance.
    float delta = old_delta * new_delta;
    Utils::atomic_add(m_squared_eval_diff, delta);

    const float final_score = nn_output->final_score;
    Utils::atomic_add(m_accumulated_black_finalscore, final_score);

    bool success = wait_update();
    if (success || m_terminal.load()) {
        const size_t o_size = nn_output->ownership.size();
        for (auto idx = size_t{0}; idx < o_size; ++idx) {
            const auto owner = nn_output->ownership[idx];
            m_accumulated_black_ownership[idx] += owner;
        }
        if (success) {
            update_done();
        }
    }
}

float UCTNode::get_mean_score(const int color) const {
    return get_final_score(color);
}

float UCTNode::get_score_utility(const int color, const float blance_score) const {
    const auto mean_score = get_mean_score(color);
    const auto score_utility = (std::tanh((mean_score - blance_score) / parameters()->score_utility_div) + 1) / 2.0f; 
    return score_utility;
}

bool Heuristic::should_be_resign(GameState &state, UCTNode *node, float threshold) {
    const int bsize = state.get_boardsize();
    const int num_moves = state.get_movenum();
    const int color = state.get_to_move();
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
        if (winrate >= threshold || lcb >= threshold) {
            return false;
        } 
    }
    return true;
}


bool Heuristic::pass_to_win(GameState &state) {
    if (state.get_last_move() == Board::PASS) {
        // const auto addition_komi = option<int>("mutil_labeled_komi");
        const auto res = state.final_score();
        const auto color = state.get_to_move();
        if (color == Board::BLACK && res > 0.f) {
            return true;
        } else if (color == Board::WHITE && res < 0.f) {
            return true;
        }
  }

  return false;
}

void UCT_Information::dump_stats(GameState &state, UCTNode *node, int cut_off) {
    const auto color = state.get_to_move();
    const auto lcblist = node->get_lcb_list(color);
    const auto parents_visits = static_cast<float>(node->get_visits());
    assert(color == node->get_color());

    // dump_ownership(state, node);
    const auto add_komi = [](const float komi, const int color) {
        if (color == Board::BLACK) {
            return 0.f - komi;
        }
        return komi;
    };

    const auto komi = state.get_komi();
    const auto root_ownership = node->get_ownership(color);
    Utils::auto_printf("Search List :\n"); 
    Utils::auto_printf("Root -> %7d (V: %5.2f%%) (S: %5.2f | %5.2f)\n",
                       node->get_visits(),
                       node->get_eval(color, false) * 100.f,
                       node->get_final_score(color)  + add_komi(komi, color),
                       std::accumulate(std::begin(root_ownership), std::end(root_ownership), add_komi(komi, color)));

    int push = 0;
    for (auto &lcb : lcblist) {
        const auto lcb_value = lcb.first > 0.0f ? lcb.first : 0.0f;
        const auto vtx = lcb.second;
    
        auto child = node->get_child(vtx);
        const auto visits = child->get_visits();
        //const auto pobability = child->get_policy();
        assert(visits != 0);
    
        const auto eval = child->get_eval(color, false);
        const auto move = state.vertex_to_string(vtx);
        const auto pv_string = move + " " + pv_to_srting(child, state);
        const auto visit_ratio = static_cast<float>(visits) / parents_visits;
        const auto final_score = child->get_final_score(color) + add_komi(komi, color);
        const auto ownership = child->get_ownership(color);
        auto owner_sum =
            std::accumulate(std::begin(ownership), std::end(ownership), add_komi(komi, color));

        Utils::auto_printf("%4s -> %7d (V: %5.2f%%) (LCB: %5.2f%%) (N: %5.2f%%) (S: %5.2f | %5.2f) PV: %s\n", 
                            move.c_str(),
                            visits,
                            eval * 100.f, 
                            lcb_value * 100.f,
                            visit_ratio * 100.f,
                            final_score, owner_sum,
                            pv_string.c_str());

        push++;
        if (push == cut_off) {
            Utils::auto_printf("     ...remain %d nodes\n", (int)lcblist.size() - cut_off);
            break;
        }
    
    }

    // tree_stats();
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

/*
size_t UCT_Information::get_memory_used() {
  return Edge::edge_tree_size + UCTNode::node_tree_size + DataBuffer::node_data_size;
}

void UCT_Information::dump_ownership(GameState &state, UCTNode *node) {
  const auto boardsize = state.board.get_boardsize();
  const auto color = state.board.get_to_move();
  const auto ownership = node->get_ownership(color);

  Utils::auto_printf("Ownership : \n");
  for (int y = 0; y < boardsize; ++y) {
    Utils::auto_printf(" ");
    for (int x = 0; x < boardsize; ++x) {
      const auto idx = state.board.get_index(x, y);
      const auto owner = (ownership[idx] + 1.0f) / 2.0f;
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


void UCT_Information::collect_nodes() {

}

*/
