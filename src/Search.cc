#include <numeric>

#include "Board.h"
#include "Evaluation.h"
#include "Search.h"
#include "UCTNode.h"
#include "Random.h"

using namespace Utils;

ThreadPool SearchPool;

Search::Search(GameState &state, Evaluation &evaluation, Trainer &trainer)
    : m_gamestate(state), m_evaluation(evaluation), m_trainer(trainer) {
  set_playout(cfg_playouts);

  int threads = cfg_uct_threads - 1;
  if (threads < 0) {
    threads = 0;
  }
  m_rootstate = m_gamestate;
  SearchPool.initialize(this, threads);
}

Search::~Search() {
  set_running(false);
  SearchPool.wait_finish();
  clear_nodes();
}

int Search::think(Search::strategy_t strategy) {

  if (strategy == strategy_t::NN_DIRECT) {
    return nn_direct_output();
  } else if (strategy == strategy_t::NN_UCT) {
    return uct_search();
  } else if (strategy == strategy_t::RANDOM) {
    return random_move();
  }

  return Board::NO_VERTEX;
}

int Search::random_move() {
  m_rootstate = m_gamestate;
  int move = 0;  

  auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
  const size_t boardsize = m_rootstate.board.get_boardsize();
  const size_t intersections = m_rootstate.board.get_intersections();
  while (true) {
    const size_t randmove = rng.randuint64() % intersections;
   /*
    if (randmove == intersections) {
    
      move = Board::PASS;
      break;
    } else {
    */
      const auto x = randmove % boardsize;
      const auto y = randmove / boardsize;
      const auto vtx = m_rootstate.board.get_vertex(x, y);
      const auto to_move = m_rootstate.board.get_to_move();
      const auto success = m_rootstate.board.is_legal(vtx, to_move);

      if (success) {
        move = vtx;
        break;
      }
   /*
    }
    */
  }
  return move;
}

int Search::nn_direct_output() {
  m_rootstate = m_gamestate;

  Evaluation::NNeval eval = m_evaluation.network_eval(m_rootstate, Network::Ensemble::NONE);

  int to_move = m_rootstate.board.get_to_move();
  int out_vertex = Board::NO_VERTEX;
  float bset_policy = std::numeric_limits<float>::lowest();

  const auto boardsize = m_rootstate.board.get_boardsize();
  const auto intersections = boardsize * boardsize;
  auto_printf(" policy out : \n");


  for (int y = 0; y < boardsize; ++y) {
    for (int x = 0; x < boardsize; ++x) {
      const auto vertex = m_rootstate.board.get_vertex(x, y);
      const auto idx = m_rootstate.board.get_index(x, y);
      if (m_rootstate.board.is_legal(vertex, to_move) && bset_policy < eval.policy[idx]) {
        bset_policy = eval.policy[idx];
        out_vertex = vertex;
      }
      auto_printf("%.5f ", eval.policy[idx]);
    }
    auto_printf("\n");
  }

  if (bset_policy < eval.policy_pass) {
    out_vertex = Board::PASS;
  }
  auto_printf(" pass : %.5f \n\n", eval.policy_pass);

  
  auto_printf("score belief out : \n");
  for (auto &s : eval.score_belief) {
    auto_printf("%.5f ", s);
  }
  auto_printf("\n\n");

  auto_printf("final score out : \n");
  auto_printf("%.5f ", eval.final_score);
  auto_printf("\n\n");


  auto_printf(" ownership out : \n");
  for (auto idx = size_t{0}; idx < intersections; idx++) {
    const auto x = idx % boardsize;
    auto_printf("%.5f ", eval.ownership[idx]);
    if (x == boardsize - 1) {
      auto_printf("\n");
    }
  }

  auto_printf("\n\n");
  for (auto idx = size_t{0}; idx < VALUE_LABELS; idx++) {
    const auto winrate = eval.multi_labeled[idx];
    auto_printf("%f ", winrate);
  }
  auto_printf("\n");

  auto_printf(" NN eval = ");
  auto_printf("%f", eval.winrate);
  auto_printf("%\n");

  return out_vertex;
}

void Search::set_playout(int playouts) {
  m_maxplayouts = playouts < MAX_PLAYOUYS ? playouts : MAX_PLAYOUYS;
  m_maxplayouts = m_maxplayouts >= 1 ? m_maxplayouts : 1;
}

float Search::get_min_psa_ratio() {
  auto v = m_playouts.load();
  if (v >= MAX_PLAYOUYS) {
    set_running(false);
    return 1.0f;
  }
  return 0.0f;
}

void Search::increment_playouts() {
  m_playouts++;
}

void Search::play_simulation(GameState &currstate, UCTNode *const node,
                             UCTNode *const root_node, SearchResult &search_result) {
  node->increment_threads();

  if (node->expandable()) {
    if (currstate.board.get_passes() >= 2) {
      search_result.from_score(currstate);
    } else {
      std::shared_ptr<NNOutputBuffer> nn_output;
      const bool had_children = node->has_children();
      const bool success = node->expend_children(m_evaluation, currstate, nn_output,
                                                 get_min_psa_ratio());
      if (!had_children && success) {
        search_result.from_nn_output(nn_output);
      }
    }
  }

  if (node->has_children() && !search_result.valid()) {
    const int color = currstate.board.get_to_move();
    auto next = node->uct_select_child(color, node == root_node);
    auto move = next->get_vertex();

    currstate.play_move(move, color);

    if (move != Board::PASS && currstate.superko()) {
      next->invalinode();
    } else {
      play_simulation(currstate, next, root_node, search_result);
    }
  }

  if (search_result.valid()) {
    auto out = search_result.nn_output();
    node->update(out);
  }

  node->decrement_threads();
}

void Search::updata_root(UCTNode *root_node) {
  std::shared_ptr<NNOutputBuffer> nn_output;
  root_node->prepare_root_node(m_evaluation, m_rootstate, nn_output);
  const auto to_move = m_rootstate.board.get_to_move();

  auto eval = nn_output->eval;

  if (to_move == Board::WHITE) {
    eval = 1.0f - eval;
  }
  eval *= 100.f;
  auto_printf("Root :\n");
  auto_printf(" NN eval = ");
  auto_printf("%f%%, ", eval);

  
  auto final_score = nn_output->final_score;
  if (to_move == Board::WHITE) {
    final_score = 0 - final_score;
  }

  auto score_belief = nn_output->score_belief;
  if (to_move == Board::WHITE) {
    score_belief = 0 - score_belief;
  }

  auto_printf(" NN final score = ");
  auto_printf("%.2f,", final_score);
  
  auto_printf(" NN score belief = ");
  auto_printf("%.2f,", score_belief);


  auto_printf(" label komi = ");
  auto_printf("%d\n", cfg_lable_komi + cfg_lable_shift);
}

bool check_release(size_t tot_sz, size_t sz, std::string name) {
  if (tot_sz != 0) {
    auto_printf("%s size = %zu\n", name.c_str(), tot_sz);
    auto_printf("%s count = %zu\n", name.c_str(), (tot_sz/sz));
  } 

  return tot_sz == 0; 
}

bool Search::is_in_time(const float max_time) {
  float seconds = m_timer.get_duration();
  if (seconds < max_time) {
    return true;
  }
  return false;
}

void Search::ponder_search() {
  prepare_uct_search();
  SearchPool.wakeup();
}

void Search::ponder_stop() {
  set_running(false);
  SearchPool.wait_finish();
  clear_nodes();
}

int Search::uct_search() {
  m_gamestate.time_clock();

  if (cfg_ponder) {
    ponder_stop();
  }

  m_rootstate = m_gamestate;
  int select_move = Board::NO_VERTEX;
  bool keep_running = true;
  bool need_resign = false;

  const float thinking_time = m_rootstate.get_thinking_time();
  auto_printf("Max thinking time : %.4f seconds\n", thinking_time);
  
  m_timer.clock();

  prepare_uct_search();
  updata_root(m_rootnode);
  
  if (thinking_time > 0.1f) {
    SearchPool.wakeup();
  }
  do {
    auto current = std::make_shared<GameState>(m_rootstate);
    auto result = SearchResult{};
    play_simulation(*current, m_rootnode, m_rootnode, result);
    if (result.valid()) {
      increment_playouts();
    }

    keep_running &= is_over_playouts();
    keep_running &= is_in_time(thinking_time);
    set_running(keep_running);
  } while (is_uct_running());

  SearchPool.wait_finish();

  const auto seconds = m_timer.get_duration();
  const auto playouts = m_playouts.load();
  auto_printf("Basic :\n");
  auto_printf(" playouts : %d\n", playouts);
  auto_printf(" spent : %2.5f (seconds)\n", seconds);
  auto_printf(" speed : %2.5f (playouts/seconds) \n", (float)playouts / seconds );
  UCT_Information::dump_stats(m_rootstate, m_rootnode);

  select_move = select_best_move();

  need_resign = Heuristic::should_be_resign(m_rootstate, m_rootnode, cfg_resign_threshold);
 
  m_trainer.gather_step(m_rootstate, *m_rootnode);
  m_rootnode->adjust_label_shift(nullptr);
  clear_nodes();
  
  assert(select_move != Board::NO_VERTEX);

  m_gamestate.recount_time(m_gamestate.board.get_to_move());

  if (need_resign) {
    select_move = Board::RESIGN;
  }

  if (cfg_ponder && select_move != Board::RESIGN) {
    m_rootstate.play_move(select_move);
    ponder_search();
  }

  return select_move;
}

int Search::select_best_move() {
  int select_move = Board::NO_VERTEX;
  const int movenum = m_rootstate.board.get_movenum();
  const int boardsize = m_rootstate.board.get_boardsize();
  const int intersections = boardsize * boardsize;

  const int div = cfg_random_move_div > 1 ? cfg_random_move_div : 1;

  cfg_random_move_cnt = intersections / div;  
  if (movenum <= cfg_random_move_cnt && cfg_random_move) {
    select_move = m_rootnode->randomize_first_proportionally(1.0f);
  }

  if (select_move != Board::NO_VERTEX) {
    return select_move;
  }

  select_move = m_rootnode->get_best_move();
  return select_move;
}

void Search::prepare_uct_search() {

  assert(m_rootnode == nullptr);
  auto root_data = std::make_shared<DataBuffer>();

  m_rootnode = new UCTNode(root_data);
  m_playouts = 0;

  set_running(true);
}

void Search::clear_nodes() {
  if (m_rootnode == nullptr) {
    return;
  }

  delete m_rootnode;
  m_rootnode = nullptr;

  bool success = true;

  success &= check_release(Edge::edge_tree_size, sizeof(Edge), "Edge");
  success &= check_release(UCTNode::node_tree_size, sizeof(UCTNode), "Node");
  success &= check_release(DataBuffer::node_data_size, sizeof(DataBuffer), "Data");

  assert(success);
}


bool Search::is_over_playouts() const {
  return m_playouts.load() < m_maxplayouts;;
}

void Search::set_running(bool is_running) {
  m_running.store(is_running);
}

bool Search::is_uct_running() {
  return m_running.load();
}

void Search::benchmark(int playouts) {

  int old_playouts = m_maxplayouts;
  set_playout(playouts);  

  int move = uct_search();
  (void) move;

  set_playout(old_playouts);
}

void ThreadPool::add_thread() {
  m_threads.emplace_back([this](){
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
    auto result = SearchResult{};
    m_search->play_simulation(*currstate, m_root, m_root, result);
    if (result.valid()) {
      m_search->increment_playouts();
    }
  } while(m_search->is_uct_running());
  m_running_theards--;
}
