#ifndef ENGINE_H_INCLUDE
#define ENGINE_H_INCLUDE

#include "config.h"
#include "GameState.h"
#include "SGFStream.h"
#include "Evaluation.h"
#include "Trainer.h"
#include "Search.h"
#include "Board.h"

#include <memory>
#include <vector>

class Engine {
public:
    using Response = std::string;

    ~Engine();

    void initialize();
    void release();
    void display();

    Response play_textmove(std::string);

    Response undo_move();

    Response showboard(int t = 0);

    Response nn_rawout();

    Response reset_boardsize(const int bsize);

    Response reset_komi(const float komi);

    Response input_features(int symmetry);

    Response think(const int color = Board::INVAL);

    Response self_play();

    Response dump_collect(std::string file = "std-output");

    Response dump_sgf(std::string file = "std-output");

    Response set_playouts(const int p);
 
    Response clear_board();

    Response misc_features();

    Response nn_batchmark(const int times);

    Response clear_cache();

    Response random_playmove();

    const GameState& get_state() const;

private:
    std::shared_ptr<Evaluation> m_evaluation{nullptr};
    std::shared_ptr<Trainer> m_trainer{nullptr};
    std::shared_ptr<Search> m_search{nullptr};

    std::vector<std::shared_ptr<GameState>> m_states;
    std::shared_ptr<GameState> m_state{nullptr};
    size_t default_id{0};

};


#endif
