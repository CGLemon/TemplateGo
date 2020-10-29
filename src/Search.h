#ifndef SEARCH_H_INCLUDE
#define SEARCH_H_INCLUDE

#include <memory>
#include <functional>

#include "SearchParameters.h"
#include "ThreadPool.h"
#include "Search.h"
#include "Evaluation.h"
#include "GameState.h"
#include "UCTNode.h"
#include "Trainer.h"
#include "Utils.h"
#include "config.h"

class SearchResult {
public:
    SearchResult() = default;
    bool valid() const { return m_nn_outout != nullptr; }
    std::shared_ptr<NNOutput> nn_output() const { return m_nn_outout; }

    void from_nn_output(std::shared_ptr<NNOutput> nn_outout) { 
        m_nn_outout = nn_outout;
    }

    void from_score(GameState &state) {
        m_nn_outout = std::make_shared<NNOutput>();

        const auto board_score = state.final_score();
        const auto komi = state.get_komi();

        if (board_score > 0.0f) {
            m_nn_outout->eval = 1.0f;
        } else if (board_score < 0.0f) {
            m_nn_outout->eval = 0.0f;
        } else {
            m_nn_outout->eval = 0.5f;
        }

        m_nn_outout->final_score = board_score + komi;
    
        const auto ownership = state.board.get_ownership();
        const auto o_size = ownership.size();
        m_nn_outout->ownership = std::array<float, NUM_INTERSECTIONS>{};
        m_nn_outout->ownership.fill(0.0f);

        for (auto idx = size_t{0}; idx < o_size; ++idx) {
            const auto owner =  ownership[idx];
            if (owner == Board::BLACK) {
                m_nn_outout->ownership[idx] = 1.0f;
            } else if (owner == Board::WHITE) {
                m_nn_outout->ownership[idx] = -1.0f;
            }
        }
    }

private:
    std::shared_ptr<NNOutput> m_nn_outout{nullptr};

};

class Search {
public:
    static constexpr int MAX_PLAYOUYS = 150000;
    Search() = delete;
    Search(GameState &state, Evaluation &evaluation, Trainer &trainer);
    ~Search();

    enum class strategy_t { RANDOM, NN_DIRECT, NN_UCT };
    int think(strategy_t = strategy_t::NN_UCT);

    GameState m_rootstate;
    UCTNode *m_rootnode{nullptr};

    void prepare_uct_search();

private:
    int nn_direct_output();
    int random_move(bool allow_pass);

    // About UCT search
    bool is_in_time(const float max_time);
    void play_simulation(GameState &currstate, UCTNode *const node,
                       UCTNode *const root_node, SearchResult &search_result);

    void updata_root(UCTNode *root_node);
    void set_playout(int playouts);
    bool is_stop_uct_search() const;

    float get_min_psa_ratio();
    void increment_playouts();
    bool is_uct_running();

    void clear_nodes();
    int select_best_move();

    bool is_over_playouts() const;
    int uct_search();
    void set_running(bool);

    GameState & m_gamestate;
    Evaluation & m_evaluation;
    Trainer & m_trainer;
    std::unique_ptr<ThreadGroup<void>> m_threadGroup{nullptr};

    int m_maxplayouts;
    std::atomic<bool> m_running;
    std::atomic<int> m_playouts;
    Timer m_timer;
    std::shared_ptr<SearchParameters> m_parameters{nullptr};
};
#endif
