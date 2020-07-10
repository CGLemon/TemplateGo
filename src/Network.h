#ifndef NETWORK_H_INCLUDE
#define NETWORK_H_INCLUDE

#include <cassert>

#include "Board.h"
#include "CacheTable.h"
#include "GameState.h"
#include "LZ/LZNetParameters.h"
#include "LZ/LZModel.h"
#include "LZ/LZCPUbackend.h"

static constexpr int NUM_SYMMETRIES = Board::NUM_SYMMETRIES;
static constexpr int IDENTITY_SYMMETRY = Board::IDENTITY_SYMMETRY;

class Network {
public:
  enum class Networkfile_t { LEELAZ };

  enum Ensemble { DIRECT, RANDOM_SYMMETRY, AVERAGE };

  using Netresult = NNResult;
  using PolicyVertexPair = std::pair<float, int>;

  void initialize(int playouts, const std::string &weightsfile,
                  Networkfile_t file_type = Networkfile_t::LEELAZ);


  Netresult get_output(const GameState *const state, const Ensemble ensemble,
                       const int symmetry = -1, const bool read_cache = true,
                       const bool write_cache = true,
                       const bool force_selfcheck = false);

  static std::pair<int, int> get_intersections_pair(int idx, int boradsize);

private:
  void init_leelaz_batchnorm_weights();


  bool probe_cache(const GameState *const state, Network::Netresult &result);

  

  Netresult get_output_internal(const GameState *const state,
                                const int symmetry);


  CacheTable<NNResult> m_nncache;

  std::unique_ptr<LZ::NNpipe> m_lz_forward;
  std::shared_ptr<LZ::LZModel::ForwardPipeWeights> m_lz_weights;



};



#endif
