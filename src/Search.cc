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

    set_playout(option<int>("playouts"));
    int threads = option<int>("threads") - 1;
    if (threads < 0) {
        threads = 0;
    }
    m_rootstate = m_gamestate;
    SearchPool.initialize(threads);
    m_threadGroup = std::make_unique<ThreadGroup<void>>(SearchPool);
    m_parameters = std::make_shared<SearchParameters>();
}


int Search::think(Search::strategy_t strategy) {

    if (strategy == strategy_t::NN_DIRECT) {
        return nn_direct_output();
    } else if (strategy == strategy_t::NN_UCT) {
        return uct_search();
    } else if (strategy == strategy_t::RANDOM) {
        return random_move(true);
    }

    return Board::NO_VERTEX;
}
int Search::nn_direct_output() {
    m_rootstate = m_gamestate;

    Evaluation::NNeval eval = m_evaluation.network_eval(m_rootstate, Network::Ensemble::NONE);

    int to_move = m_rootstate.get_to_move();
    int out_vertex = Board::NO_VERTEX;
    float bset_policy = std::numeric_limits<float>::lowest();
    const auto boardsize = m_rootstate.get_boardsize();

    for (int y = 0; y < boardsize; ++y) {
        for (int x = 0; x < boardsize; ++x) {
            const auto vertex = m_rootstate.get_vertex(x, y);
            const auto idx = m_rootstate.get_index(x, y);
            if (m_rootstate.is_legal(vertex, to_move) && bset_policy < eval.policy[idx]) {
                bset_policy = eval.policy[idx];
                out_vertex = vertex;
            }
        }
    }

    if (bset_policy < eval.policy_pass) {
        out_vertex = Board::PASS;
    }

    return out_vertex;
}



int Search::random_move(bool allow_pass) {
    m_rootstate = m_gamestate;
    int move = 0;  

    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    const size_t boardsize = m_rootstate.get_boardsize();
    const size_t intersections = m_rootstate.get_intersections();
    while (true) {
        const size_t randmove = rng.randuint64() % intersections;
   
        if (randmove == intersections && allow_pass) {
            move = Board::PASS;
            break;
        } else {
            const auto x = randmove % boardsize;
            const auto y = randmove / boardsize;
            const auto vtx = m_rootstate.get_vertex(x, y);
            const auto to_move = m_rootstate.get_to_move();
            const auto success = m_rootstate.is_legal(vtx, to_move);

            if (success) {
                move = vtx;
                break;
            }   
        }
    }
    return move;
}



// UCT search
int Search::uct_search() {

    const auto uct_worker = [&](){
        do {
            auto currstate = std::make_unique<GameState>(m_rootstate);
            auto result = SearchResult{};
            play_simulation(*currstate, m_rootnode, m_rootnode, result);
            if (result.valid()) {
                increment_playouts();
            }
        } while(is_uct_running());
    };

    // Start to clock.
    m_gamestate.time_clock();
    m_timer.clock();

    const float thinking_time = m_rootstate.get_thinking_time();
    auto_printf("Max thinking time : %.4f seconds\n", thinking_time);

    if (option<bool>("ponder")) {
        // If pondering, we clear nodes first.
        // The nodes are not necessary.
        m_threadGroup->wait_all();
        clear_nodes();
    }

    auto_printf("Initializing...\n");

    // Initialize before searching.
    m_rootstate = m_gamestate;
    int select_move = Board::NO_VERTEX;
    bool keep_running = true;
    bool need_resign = false;
    prepare_uct_search();
    updata_root(m_rootnode);

    auto_printf("Start searching...\n");
    if (thinking_time > 0.1f) {
        // Wake up the threads need some time.
        // If thinking time is too small, we do not allow
        // to wake up the threads
        m_threadGroup->fill_tasks(uct_worker);
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

    m_threadGroup->wait_all();

    const auto seconds = m_timer.get_duration();
    const auto playouts = m_playouts.load();
    auto_printf("Basic :\n");
    auto_printf(" playouts : %d\n", playouts);
    auto_printf(" spent : %2.5f (seconds)\n", seconds);
    auto_printf(" speed : %2.5f (playouts/seconds) \n", (float)playouts / seconds );
    UCT_Information::dump_stats(m_rootstate, m_rootnode);

    select_move = select_best_move();

    need_resign = Heuristic::should_be_resign(m_rootstate, m_rootnode, option<float>("resigned_threshold"));
 
    m_trainer.gather_step(m_rootstate, *m_rootnode);
    clear_nodes();
  
    assert(select_move != Board::NO_VERTEX);

    m_gamestate.recount_time(m_gamestate.get_to_move());

    if (need_resign) {
        select_move = Board::RESIGN;
    }

    if (option<bool>("ponder") && select_move != Board::RESIGN) {
        m_rootstate.play_move(select_move);
        m_threadGroup->fill_tasks(uct_worker);
    }

    return select_move;

}

void Search::prepare_uct_search() {
    auto_printf("preparing uct search...\n");
    assert(m_rootnode == nullptr);
    auto root_data = std::make_shared<UCTData>();
    root_data->parameters = m_parameters;

    m_rootnode = new UCTNode(root_data);
    m_playouts.store(0);

    set_running(true);
}

void Search::updata_root(UCTNode *root_node) {
    auto_printf("updating root...\n");
    std::shared_ptr<NNOutput> nn_output;
    root_node->prepare_root_node(m_evaluation, m_rootstate, nn_output);
    const auto to_move = m_rootstate.get_to_move();
    auto eval = nn_output->eval;

    if (to_move == Board::WHITE) {
        eval = 1.0f - eval;
    }

    eval *= 100.f;
    auto_printf("Root :\n");
    auto_printf(" NN eval = ");
    auto_printf("%f%% \n", eval);

  
    auto final_score = nn_output->final_score;
    if (to_move == Board::WHITE) {
        final_score = 0 - final_score;
    }

    auto_printf(" NN final score = ");
    auto_printf("%.2f\n", final_score);

    auto_printf(" label komi = ");
    auto_printf("%d\n", option<int>("mutil_labeled_komi"));
}

void Search::set_running(bool is_running) {
    m_running.store(is_running);
}

bool Search::is_uct_running() {
    return m_running.load();
}

bool Search::is_over_playouts() const {
    return m_playouts.load() < m_maxplayouts;;
}

int Search::select_best_move() {
    int select_move = Board::NO_VERTEX;
    const int movenum = m_rootstate.get_movenum();
    const int intersections = m_rootstate.get_intersections();
    const int div = option<int>("random_move_div") > 1 ? option<int>("random_move_div") : 1;
    const auto random_move_cnt = intersections / div;  

    if (movenum <= random_move_cnt && option<bool>("random_move")) {
        select_move = m_rootnode->randomize_first_proportionally(1.0f);
    }

    if (select_move != Board::NO_VERTEX) {
        return select_move;
    }

    select_move = m_rootnode->get_best_move();
    return select_move;
}


void Search::clear_nodes() {
    if (m_rootnode == nullptr) {
        return;
    }

    delete m_rootnode;
    m_rootnode = nullptr;

    //bool success = true;

    //success &= check_release(Edge::edge_tree_size, sizeof(Edge), "Edge");
    //success &= check_release(UCTNode::node_tree_size, sizeof(UCTNode), "Node");
    //success &= check_release(DataBuffer::node_data_size, sizeof(DataBuffer), "Data");

    //assert(success);
}

void Search::play_simulation(GameState &currstate, UCTNode *const node,
                             UCTNode *const root_node, SearchResult &search_result) {
    node->increment_threads();
    if (node->expandable()) {
        if (currstate.get_passes() >= 2) {
            search_result.from_score(currstate);
            node->from_nn_output(search_result.nn_output());
        } else {
            std::shared_ptr<NNOutput> nn_output;
            const bool had_children = node->has_children();
            const bool success = node->expend_children(m_evaluation, currstate, nn_output,
                                                       get_min_psa_ratio());
            if (!had_children && success) {
                search_result.from_nn_output(nn_output);
            }
        }
    }

    if (node->has_children() && !search_result.valid()) {
        const int color = currstate.get_to_move();
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

float Search::get_min_psa_ratio() {
    auto v = m_playouts.load();
    if (v >= MAX_PLAYOUYS) {
    set_running(false);
        return 1.0f;
    }
    return 0.0f;
}

void Search::set_playout(int playouts) {
    m_maxplayouts = playouts < MAX_PLAYOUYS ? playouts : MAX_PLAYOUYS;
    m_maxplayouts = m_maxplayouts >= 1 ? m_maxplayouts : 1;
}

void Search::increment_playouts() {
    m_playouts++;
}

bool Search::is_in_time(const float max_time) {
    float seconds = m_timer.get_duration();
    if (seconds < max_time) {
        return true;
    }
    return false;
}

Search::~Search() {
    set_running(false);
    clear_nodes();
}
