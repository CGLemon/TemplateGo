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

class SearchResult {
public:
  SearchResult() = default;
  bool valid() const { return m_valid; }
  float eval() const { return m_eval; }
  static SearchResult from_eval(float eval) { return SearchResult(eval); }
  static SearchResult from_score(float board_score) {
    if (board_score > (0.0f+m_error)) {
      return SearchResult(1.0f);
    } else if (board_score < (0.0f-m_error)) {
      return SearchResult(0.0f);
    } else {
      return SearchResult(0.5f);
    }
  }
private:
  explicit SearchResult(float eval) : m_valid(true), m_eval(eval) {}
  bool m_valid{false};
  float m_eval{0.0f};
  static constexpr float m_error{1e-4f};
};


class Search {
public:
  enum class strategy_t { NN_DIRECT, NN_UCT };

  static constexpr int MAX_PLAYOUYS = 150000;

  Search(GameState &state, Evaluation &evaluation);
  int think(strategy_t);

  int nn_direct_output();
  int uct_search();

  SearchResult play_simulation(GameState &currstate, UCTNode *const node,
                               UCTNode *const root_node);

  void updata_root(std::shared_ptr<UCTNode> root_node);
  void set_playout(int playouts);
  bool is_stop_uct_search() const;

  float get_min_psa_ratio() const;
  void increment_playouts();
  bool is_uct_running();

  void benchmark(int playouts);

  GameState & m_rootstate;
  UCTNode * m_rootnode;

  void prepare_uct_search();
  bool is_over_playouts() const;
  void set_running(bool);

private:

  Evaluation & m_evaluation;
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
