#ifndef TRAINER_H_INCLUDE
#define TRAINER_H_INCLUDE

#include <vector>
#include <list>
#include <iostream>
#include <string>

#include "GameState.h"
#include "UCTNode.h"

class Trainer {
public:
    void gather_step(GameState &state, UCTNode &node);
    void gather_step(GameState &state, const int vtx);
    void gather_winner(GameState &state);

    void clear_game_steps();

    void dump_memory() const;
    void save_data(std::string &filename, bool append = false);
    void data_stream(std::ostream &out);

private:
    struct Step {
        std::vector<char> input_planes;
        std::vector<float> input_features; 
        std::vector<float> probabilities;
        std::vector<float> opponent_probabilities;
        std::vector<int> ownership;

        int final_score;  // Actually, This is score on board. Do Not add komi.

        // int final_score_idx;
        // std::vector<float> results;

        float result;
        int to_move;
        int board_size;

        float current_komi;
        void step_stream(std::ostream &out);
    };

    void scatch_step(GameState &state, Step &step) const;
    void push_game_step(Step &step);
    void adjust_game_steps(size_t size);

    std::list<Step> game_steps;
};
#endif
