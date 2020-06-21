#include <numeric>

#include "Board.h"
#include "Evaluation.h"
#include "Search.h"
#include "UCTNode.h"
#include "Utils.h"
#include "cfg.h"

using namespace Utils;

ThreadPool SearchPool;

Search::Search(GameState &state, Evaluation &evaluation)
    : m_rootstate(state), m_evaluation(evaluation) {
  set_playout(cfg_playouts);
  SearchPool.initialize(this, cfg_uct_threads);
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

  auto_printf("policy out : \n");

  for (int idx = 0; idx < NUM_INTERSECTIONS; idx++) {
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

    auto_printf("%.5f ", eval.policy[idx]);
    if (x == BOARD_SIZE - 1) {
      auto_printf("\n");
    }
  }
  if (most_policy < eval.policy_pass) {
    out_vertex = Board::PASS;
  }
  auto_printf("pass : %.5f \n", eval.policy_pass);

  auto_printf("NN eval = ");
  auto_printf("%f\n", eval.winrate[0]);
  auto_printf("%");
  return out_vertex;
}

void Search::set_playout(int playouts) {
  m_maxplayouts = playouts < MAX_PLAYOUYS ? playouts : MAX_PLAYOUYS;
  m_maxplayouts = m_maxplayouts >= 1 ? m_maxplayouts : 1;
}

float Search::get_min_psa_ratio() const { return 0.0f; }

void Search::increment_playouts() { m_playouts++; }

SearchResult Search::play_simulation(GameState &currstate, UCTNode *const node,
                                     UCTNode *const root_node) {
  auto result = SearchResult{};
  node->increment_virtual_loss();

  if (node->expandable()) {
    if (currstate.board.get_passes() >= 2) {
      float score = currstate.final_score();
      result = SearchResult::from_score(score);
    } else {
      float eval;
      const bool had_children = node->has_children();
      const bool success = node->expend_children(m_evaluation, currstate, eval,
                                                 get_min_psa_ratio());
      if (!had_children && success) {
        result = SearchResult::from_eval(eval);
      }
    }
  }

  if (node->has_children() && !result.valid()) {
    const int color = currstate.board.get_to_move();
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

void Search::updata_root(std::shared_ptr<UCTNode> root_node) {
  float eval = root_node->prepare_root_node(m_evaluation, m_rootstate);

  if (m_rootstate.board.get_to_move() == Board::WHITE) {
    eval = 1.0f - eval;
  }
  auto_printf("NN eval = ");
  auto_printf("%f", eval);
  auto_printf("%\n");
}

bool check_release(size_t tot_sz, size_t sz, std::string name) {
  if (tot_sz != 0) {
    auto_printf("%s size = %zu\n", name.c_str(), tot_sz);
    auto_printf("%s count = %zu\n", name.c_str(), (tot_sz/sz));
  } 

  return (tot_sz == 0); 
}

int Search::uct_search() {
  int select_move;
  {
    auto root_data = std::make_shared<DataBuffer>();
    auto root_node = std::make_shared<UCTNode>(root_data);

    m_rootnode = root_node.get();
    updata_root(root_node);
    m_playouts = 0;
  
    SearchPool.wakeup();

    do {
      auto current = std::make_shared<GameState>(m_rootstate);
      auto result = play_simulation(*current, m_rootnode, m_rootnode);
      if (result.valid()) {
        increment_playouts();
      }
    } while (is_uct_running());

    SearchPool.wait_finish();

    select_move = root_node->get_best_move();
    UCT_Information::dump_stats(m_rootnode, m_rootstate);
  }
  
  bool success = check_release(Edge::edge_tree_size, sizeof(Edge), "Edge");
  success &= check_release(UCTNode::node_tree_size, sizeof(UCTNode), "Node");
  success &= check_release(DataBuffer::node_data_size, sizeof(DataBuffer), "Data");

  assert(success);

  return select_move;
}



bool Search::is_uct_running() {
  bool keep_running = m_playouts.load() < m_maxplayouts;
  return keep_running;
}

void Search::benchmark(int playouts) {

  set_playout(playouts);  
  Timer timer;

  int move = uct_search();
  float seconds = timer.get_duration();

  auto_printf("playouts : %d\n", m_playouts.load());
  auto_printf("time : %2.5f seconds\n", seconds);

  set_playout(cfg_playouts);
}

void ThreadPool::add_thread() {
  m_threads.emplace_back([this] {
    while(true) {
      {
        m_searching = false;
        std::unique_lock<std::mutex> lock(m_mutex);
        m_condvar.wait(lock, [this]{ return m_searching; });
        if (m_exit) {
          return;
        }
      }
      worker();
    }
  });
}

void ThreadPool::initialize(Search * search, size_t threads) {
  m_search = search;
  for (size_t i = 0; i < threads; i++) {
    add_thread();
  }
}

void ThreadPool::wakeup() {
  m_searching = true;
  m_condvar.notify_all();
}

void ThreadPool::quit() {
  m_exit = true;
  wakeup();
  for (auto &t : m_threads) {
    t.join();
  } 
}

void ThreadPool::wait_finish() {
  while(m_running_theards.load() != 0) {
    ;
  }
}


void ThreadPool::worker() {
  m_running_theards++;
  auto m_root = m_search->m_rootnode;
  do {
    auto currstate = std::make_unique<GameState>(m_search->m_rootstate);
    auto result = m_search->play_simulation(*currstate, m_root, m_root);
    if (result.valid()) {
      m_search->increment_playouts();
    }
  } while(m_search->is_uct_running());
  m_running_theards--;
}
