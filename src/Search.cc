
#include <numeric>

#include "cfg.h"
#include "UCTNode.h"
#include "Utils.h"
#include "Search.h"
#include "Board.h"
#include "Evaluation.h"

using namespace Utils;

Search::Search(GameState& state, Evaluation & evaluation)
				 : m_rootstate(state), m_evaluation(evaluation) {
	set_playout(cfg_playouts);
}

int Search::think(Search::strategy_t strategy) {

	if (strategy == strategy_t::NN_DIRECT) {
		return nn_direct_output();
	} else if (strategy == strategy_t::NN_UCT) {
		return uct_search();
	}

	return Board::NO_VERTEX;
}

int Search::nn_direct_output() {
	Evaluation::NNeval eval = m_evaluation.network_eval(m_rootstate);

	int to_move = m_rootstate.board.get_to_move();
	int out_vertex = Board::NO_VERTEX;	
	float most_policy = std::numeric_limits<float>::lowest(); 
	for (int idx = 0; idx < NUM_INTERSECTIONS; idx++) {
		//const int x = idx % BOARD_SIZE;
        //const int y = idx / BOARD_SIZE;
		const auto idx_pair = Network::get_intersections_pair(idx, BOARD_SIZE);
		const int x = idx_pair.first;
        const int y = idx_pair.second;
        const int vertex = m_rootstate.board.get_vertex(x, y);

        if (m_rootstate.board.is_legal(vertex, to_move)) {
            if (most_policy < eval.policy[idx]) {
				most_policy = eval.policy[idx];
				out_vertex = vertex;
			}
        }
	}
	if (most_policy < eval.policy_pass) {
		out_vertex = Board::PASS;
	}



	auto_printf("NN eval = ");
	auto_printf("%f\n", eval.winrate[0]);
	auto_printf("%");
	return out_vertex;
}

void Search::set_playout(int playouts) {
	m_maxplayouts = playouts <  MAX_PLAYOUYS ? playouts : MAX_PLAYOUYS;
	m_maxplayouts = m_maxplayouts >= 1 ? m_maxplayouts : 1;
}

float Search::get_min_psa_ratio() const {
	return 0.0f;
}

void Search::increment_playouts() {
    m_playouts++;
}

SearchResult Search::play_simulation(GameState & currstate,
                                        UCTNode* const node,
										UCTNode* const root_node) {
    const auto color = currstate.board.get_to_move();
    auto result = SearchResult{};

    node->increment_virtual_loss();

    if (node->expandable()) {
        if (currstate.board.get_passes() >= 2) {
            float score = currstate.final_score();
            result = SearchResult::from_score(score);
        } else {
            float eval;
            const bool had_children = node->has_children();
            const bool success = node->expend_children(m_evaluation, currstate, eval, get_min_psa_ratio());
            if (!had_children && success) {
                result = SearchResult::from_eval(eval);
            }
        }
    }
	
    if (node->has_children() && !result.valid()) {
        auto next = node->uct_select_child(color, node == root_node);
        auto move = next->get_vertex();

        currstate.play_move(move, color);
		currstate.exchange_to_move();
        if (move != Board::PASS && currstate.superko()) {
            next->invalinode();
        } else {
            result = play_simulation(currstate, next, root_node);
        }
    }

    if (result.valid()) {
        node->update(result.eval());
    }

    node->decrement_virtual_loss();
	
    return result;
}

bool Search::is_stop_uct_search() const {
	return m_playouts.load() <= m_maxplayouts;
}


void Search::updata_root(std::shared_ptr<UCTNode> root_node) {
	float eval = root_node -> prepare_root_node(m_evaluation, m_rootstate);

	if (m_rootstate.board.get_to_move() == Board::WHITE) {
		eval = 1.0f - eval;
	} 	
	auto_printf("NN eval = ");
	auto_printf("%f\n", eval);
	auto_printf("%");
}

int Search::uct_search() {
	int select_move;
	{
		int color = m_rootstate.board.get_to_move();

		auto root_node = std::make_shared<UCTNode>(1.0f, Board::PASS, 0.0f);
		auto current = std::make_shared<GameState>(m_rootstate);

		updata_root(root_node);
		
		do {
			increment_playouts();
			play_simulation(*current, root_node.get(), root_node.get());
			m_running = is_stop_uct_search();
		} while(m_running);
		
		select_move = root_node->get_most_visits_move();
	}

	assert(Edge::edge_tree_size == 0);
	assert(Edge::edge_node_count == 0);
	assert(UCTNode::node_tree_size == 0);
	assert(UCTNode::node_node_count == 0);

	return select_move;
}
