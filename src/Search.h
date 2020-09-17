#ifndef SEARCH_H_INCLUDE
#define SEARCH_H_INCLUDE

#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <functional>

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
  std::shared_ptr<NNOutputBuffer> nn_output() const { return m_nn_outout; }

  void from_nn_output(std::shared_ptr<NNOutputBuffer> nn_outout) { 
    m_nn_outout = nn_outout;
  }

  void from_score(GameState &state) {
    m_nn_outout = std::make_shared<NNOutputBuffer>();
    const auto addtion_komi = cfg_lable_komi + cfg_lable_shift;
    const auto board_score = state.final_score(addtion_komi);

    if (board_score > 0.0f) {
      m_nn_outout->eval = 1.0f;
    } else if (board_score < 0.0f) {
      m_nn_outout->eval = 0.0f;
    } else {
      m_nn_outout->eval = 0.5f;
    }

    m_nn_outout->score_belief = board_score;

    m_nn_outout->final_score = board_score;
    
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
  std::shared_ptr<NNOutputBuffer> m_nn_outout{nullptr};

};



class Search {
public:
  Search() = delete;

  enum class strategy_t { RANDOM, NN_DIRECT, NN_UCT };

  static constexpr int MAX_PLAYOUYS = 150000;

  Search(GameState &state, Evaluation &evaluation, Trainer &trainer);
  ~Search();

  int think(strategy_t);

  int random_move();
  int nn_direct_output();
  int uct_search();

  void play_simulation(GameState &currstate, UCTNode *const node,
                       UCTNode *const root_node, SearchResult &search_result);

  void updata_root(UCTNode *root_node);
  void set_playout(int playouts);
  bool is_stop_uct_search() const;

  float get_min_psa_ratio();
  void increment_playouts();
  bool is_uct_running();

  void benchmark(int playouts);

  UCTNode * m_rootnode{nullptr};

  void prepare_uct_search();
  bool is_over_playouts() const;
  void set_running(bool);
  void clear_nodes();

  GameState m_rootstate;

  bool is_in_time(const float max_time);

private:
  void ponder_search();
  void ponder_stop();

  int select_best_move();

  Trainer & m_trainer;
  Evaluation & m_evaluation;
  GameState & m_gamestate;

  Timer m_timer;

  int m_maxplayouts;
  std::atomic<bool> m_running;
  std::atomic<int> m_playouts;
};


class ThreadPool {
  Search *m_search;
  std::condition_variable m_condvar;
  std::mutex m_mutex;

  void worker();
  void add_thread();

  std::vector<std::thread> m_threads;
  std::atomic<size_t> m_running_theards{0};
  bool m_searching{false};
  bool m_exit{false};

public:
  ThreadPool() = default;
  ThreadPool(Search *search, size_t threads)  
                    {initialize(search, threads); };
  ~ThreadPool() { quit(); }

  void initialize(Search *search, size_t threads);
  void wakeup();
  void quit();
  void wait_finish();
};

extern ThreadPool SearchPool;

#endif
