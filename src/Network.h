#ifndef NETWORK_H_INCLUDE
#define NETWORK_H_INCLUDE

#include <cassert>

#include "Model.h"
#include "Board.h"
#include "CacheTable.h"
#include "GameState.h"
class Network {
public:
  enum Ensemble { NONE, DIRECT, RANDOM_SYMMETRY, AVERAGE };

  using Netresult = NNResult;
  using PolicyVertexPair = std::pair<float, int>;

  void initialize(int playouts, const std::string &weightsfile);

  void reload_weights(const std::string &weightsfile);

  Netresult get_output(const GameState *const state,
                       const Ensemble ensemble,
                       const int symmetry = -1,
                       const bool read_cache = true,
                       const bool write_cache = true);

  static std::pair<int, int> get_intersections_pair(int idx, int boradsize);

  void clear_cache();

  void release_nn();

private:
  static constexpr int NUM_SYMMETRIES = Board::NUM_SYMMETRIES;
  static constexpr int IDENTITY_SYMMETRY = Board::IDENTITY_SYMMETRY;

  bool probe_cache(const GameState *const state,
                   Network::Netresult &result,
                   const int symmetry = -1);

  Netresult get_output_internal(const GameState *const state,
                                const int symmetry);
  
  Netresult get_output_form_cache(const GameState *const state);

  CacheTable<NNResult> m_nncache;

  std::unique_ptr<Model::NNpipe> m_forward;
  std::shared_ptr<Model::NNweights> m_weights;

};



#endif
