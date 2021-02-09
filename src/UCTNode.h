#ifndef UCTNODE_H_INCLUDE
#define UCTNODE_H_INCLUDE

#include "SearchParameters.h"
#include "Evaluation.h"
#include "GameState.h"
#include "NodePointer.h"
#include "Board.h"

#include <atomic>
#include <cstdint>
#include <memory>
#include <vector>

struct UCTData {
    float policy{0.0f};
    int vertex{Board::NO_VERTEX};
    std::shared_ptr<SearchParameters> parameters{nullptr};
};

struct NNOutput {
    float eval;
    float final_score;

    std::array<float, NUM_INTERSECTIONS> ownership;
};

#define VIRTUAL_LOSS_COUNT (2)

class UCTNode {
public:
    using UCTNodePointer = NodePointer<UCTNode, UCTData>;
    UCTNode(std::shared_ptr<UCTData> data);
    ~UCTNode();

    bool expend_children(Evaluation &evaluation,
                         GameState &state,
                         std::shared_ptr<NNOutput> &nn_output,
                         const float min_psa_ratio, const bool is_root = false);

    int get_vertex() const;
    float get_policy() const;
    int get_visits() const;

    int get_color() const;
    float get_raw_evaluation(const int color) const;
    float get_accumulated_evals() const;
    float get_eval(const int color, bool use_virtual_loss = true) const;
    float get_final_score(const int color) const;
    int get_most_visits_move();
    std::array<float, NUM_INTERSECTIONS> get_ownership(const int color) const;
    const std::vector<std::shared_ptr<UCTNodePointer>>& get_children() const;
    std::vector<std::pair<float, int>> get_lcb_list(const int color);
    std::vector<std::pair<float, int>> get_winrate_list(const int color);

    int get_best_move();
    int randomize_first_proportionally(float random_temp);
    UCTNode *get_child(const int vtx);
    UCTNode *get_most_visits_child();
    UCTNode *get();
    UCTNode *uct_select_child(const int color, bool is_root);

    void increment_threads();
    void decrement_threads();
    int get_virtual_loss() const;

    void set_policy(const float policy);
    bool has_children() const;
    bool is_pruned() const;
    bool is_valid() const;
    bool is_active() const;

    bool is_expending() const;
    bool is_expended() const;
    bool expandable() const;

    void inflate_all_children();
    void prepare_root_node(Evaluation &evaluation, GameState &state, std::shared_ptr<NNOutput> &nn_output);

    void set_active(const bool active);
    void invalinode();
    void update(std::shared_ptr<NNOutput> nn_output);
    void accumulate_eval(float eval);
    bool prune_child(const int vtx);

    void from_nn_output(std::shared_ptr<NNOutput> nn_output);

private:
    std::vector<std::shared_ptr<UCTNodePointer>> m_children;
    std::shared_ptr<UCTData> m_data;
    std::shared_ptr<SearchParameters> parameters() const;
    enum Status : std::uint8_t { 
        INVALID, 
        PRUNED,
        ACTIVE
    };
    std::atomic<Status> m_status{ACTIVE};  

    int m_color{Board::INVAL};
    // Network raw output
    float m_raw_black_eval{0};
    float m_raw_black_final_score{0.0f};
    std::array<float, NUM_INTERSECTIONS> m_raw_black_ownership;

    // Network accumulated output
    std::atomic<int> m_visits{0};
    std::atomic<float> m_squared_eval_diff{1e-4f};
    std::atomic<float> m_accumulated_black_evals{0.0f};
    std::atomic<float> m_accumulated_black_finalscore{0.0f};
    std::atomic<int> m_loading_threads{0};
    std::array<float, NUM_INTERSECTIONS> m_accumulated_black_ownership;
    std::atomic<bool> m_terminal{false};

    void link_nodelist(std::vector<Network::PolicyVertexPair> &nodelist, float min_psa_ratio);
    int get_threads() const;
    float get_eval_variance(float default_var) const;
    float get_eval_lcb(const int color) const;
    float get_mean_score(const int color) const;
    float get_score_utility(const int color, const float blance_score) const;
    void dirichlet_noise(float epsilon, float alpha);
    void link_nn_output(GameState &state,
                        const Evaluation::NNeval &raw_netlist,
                        std::shared_ptr<NNOutput> &nn_output, const int color);

    enum class ExpandState : std::uint8_t { 
        INITIAL = 0,
        EXPANDING, 
        EXPANDED,
        UPDATE
    };
    std::atomic<ExpandState> m_expand_state{ExpandState::INITIAL};

    bool acquire_update();
    bool wait_update();
    void update_done();
    bool acquire_expanding();

    // EXPANDING -> DONE
    void expand_done();

    // EXPANDING -> INITIAL
    void expand_cancel();

    // wait until we are on EXPANDED state
    void wait_expanded();
};

class Heuristic {
public:
    static bool pass_to_win(GameState & state);

    static bool should_be_resign(GameState & state, UCTNode *node, float threshold);

};


class UCT_Information {
public:
  static size_t get_memory_used();
  
  static void tree_stats();

  static void dump_ownership(GameState &state, UCTNode *node); 

  static void dump_stats(GameState& state, UCTNode *node, int cut_off = -1);

  static std::string pv_to_srting(UCTNode *node, GameState& state);

  static void collect_nodes();

};

#endif
