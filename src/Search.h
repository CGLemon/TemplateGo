#ifndef SEARCH_H_INCLUDE
#define SEARCH_H_INCLUDE

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
    if (board_score > 0.0f) {
      return SearchResult(1.0f);
    } else if (board_score < 0.0f) {
      return SearchResult(0.0f);
    } else {
      return SearchResult(0.5f);
    }
  }

private:
  explicit SearchResult(float eval) : m_valid(true), m_eval(eval) {}
  bool m_valid{false};
  float m_eval{0.0f};
};

class Search {
public:
  enum class strategy_t { NN_DIRECT, NN_UCT };

  static constexpr int MAX_PLAYOUYS = 10000;

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

private:
  Evaluation &m_evaluation;
  GameState &m_rootstate;

  std::atomic<bool> m_running;
  int m_maxplayouts;
  std::atomic<int> m_playouts;
};

#endif
