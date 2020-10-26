#ifndef NETWORK_H_INCLUDE
#define NETWORK_H_INCLUDE

#include <cassert>

#include "Model.h"
#include "Board.h"
#include "Cache.h"
#include "GameState.h"
class Network {
public:
    ~Network();

    enum Ensemble { NONE, DIRECT, RANDOM_SYMMETRY, AVERAGE };

    using Netresult = NNResult;
    using PolicyVertexPair = std::pair<float, int>;

    void initialize(const int playouts, const std::string &weightsfile);

    void reload_weights(const std::string &weightsfile);

    Netresult get_output(const GameState *const state,
                         const Ensemble ensemble,
                         const int symmetry = -1,
                         const bool read_cache = true,
                         const bool write_cache = true);

    void clear_cache();

    void release_nn();

    void set_playouts(const int playouts);


private:
    static constexpr int NUM_SYMMETRIES = Board::NUM_SYMMETRIES;
    static constexpr int IDENTITY_SYMMETRY = Board::IDENTITY_SYMMETRY;

    bool probe_cache(const GameState *const state,
                     Network::Netresult &result,
                     const int symmetry = -1);

    void dummy_forward(std::vector<float> &policy,
                       std::vector<float> &ownership,
                       std::vector<float> &final_score,
                       std::vector<float> &values);


    Netresult get_output_internal(const GameState *const state,
                                  const int symmetry);
  
    Netresult get_output_form_cache(const GameState *const state);

    Cache<NNResult> m_cache;

    std::unique_ptr<Model::NNpipe> m_forward;
    std::shared_ptr<Model::NNweights> m_weights;

};



#endif
