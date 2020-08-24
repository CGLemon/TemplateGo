
#include "Network.h"

#ifdef USE_ZLIB
#include "zlib.h"
#endif

#include "CPUbackend.h"
#include "Board.h"
#include "GameState.h"
#include "Random.h"
#include "Utils.h"
#include "Blas.h"
#include "cfg.h"
#include "config.h"

#ifdef USE_CUDA
#include "CUDAbackend.h"
#endif

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#include <algorithm>
#include <array>
#include <cassert>
#include <memory>
#include <string>
#include <vector>

using namespace Utils;

void Network::initialize(int playouts, const std::string &weightsfile) {

#ifndef __APPLE__
#ifdef USE_OPENBLAS
  openblas_set_num_threads(1);
  auto_printf("BLAS Core: %s\n", openblas_get_corename());
#endif
#ifdef USE_MKL
  // mkl_set_threading_layer(MKL_THREADING_SEQUENTIAL);
  mkl_set_num_threads(1);
  MKLVersion Version;
  mkl_get_version(&Version);
  auto_printf("BLAS core: MKL %s\n", Version.Processor);
#endif
#endif

#ifdef USE_EIGEN
  auto_printf("BLAS Core: built-in Eigen %d.%d.%d library.\n",
              EIGEN_WORLD_VERSION, EIGEN_MAJOR_VERSION, EIGEN_MINOR_VERSION);
#endif
  const size_t cache_size = cfg_cache_ratio * NUM_INTERSECTIONS * 
                            cfg_cache_moves * playouts;
  m_nncache.resize(cache_size);

#ifdef USE_CUDA
  using backend = CUDAbackend;
#else
  using backend = CPUbackend;
#endif


  m_weights = std::make_shared<Model::NNweights>();
  Model::loader(weightsfile, m_weights);

  m_forward = std::make_unique<backend>();
  m_forward->initialize(m_weights);

  if (m_weights->loaded) {
    auto_printf("Weights are pushed down\n");
  }

  m_weights.reset();
  m_weights = nullptr;
}

void Network::reload_weights(const std::string &weightsfile) {
  if (m_weights != nullptr) {
    return;
  }
  m_weights = std::make_shared<Model::NNweights>();
  Model::loader(weightsfile, m_weights);

  m_forward->reload(m_weights);

  if (m_weights->loaded) {
    auto_printf("Weights are pushed down\n");
  }
  m_weights.reset();
  m_weights = nullptr;
}

// TODO: probe_cache 翻轉搜尋?
bool Network::probe_cache(const GameState *const state,
                          Network::Netresult &result) {
  if (m_nncache.lookup(state->board.get_hash(), result)) {
    return true;
  }
  // If we are not generating a self-play game, try to find
  // symmetries if we are in the early opening.
  /*
if (!cfg_noise && !cfg_random_cnt
  && state->get_movenum()
     < (state->get_timecontrol().opening_moves(BOARD_SIZE) / 2)) {
  for (auto sym = 0; sym < Network::NUM_SYMMETRIES; ++sym) {
      if (sym == Network::IDENTITY_SYMMETRY) {
          continue;
      }
      const auto hash = state->get_symmetry_hash(sym);
      if (m_nncache.lookup(hash, result)) {
          decltype(result.policy) corrected_policy;
          for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; ++idx) {
              const auto sym_idx = symmetry_nn_idx_table[sym][idx];
              corrected_policy[idx] = result.policy[sym_idx];
          }
          result.policy = std::move(corrected_policy);
          return true;
      }
  }
}
  */

  return false;
}

std::pair<int, int> Network::get_intersections_pair(int idx, int boradsize) {
  const int x = idx % boradsize;
  const int y = idx / boradsize;
  return {x, y};
}


Network::Netresult Network::get_output_internal(const GameState *const state,
                                                const int symmetry) {
  assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

  auto policy_out = std::vector<float>(POTENTIAL_MOVES);
  auto finalscore_out = std::vector<float>(OUTPUTS_FINALSCORE * NUM_INTERSECTIONS);
  auto ownership_out = std::vector<float>(OUTPUTS_OWNERSHIP * NUM_INTERSECTIONS);
  auto winrate_out = std::vector<float>(VALUE_LABELS);


  const auto boardsize = state->board.get_boardsize();
  const auto input_planes = Model::gather_planes(state, symmetry);
  const auto input_features = Model::gather_features(state);

  m_forward->forward(boardsize, input_planes, input_features,
                     policy_out, finalscore_out, ownership_out, winrate_out);

  const auto result = Model::get_result(state,
                                        policy_out,
                                        finalscore_out,
                                        ownership_out,
                                        winrate_out,
                                        cfg_softmax_temp, symmetry);

  return result;
}

Network::Netresult
Network::get_output(const GameState *const state, const Ensemble ensemble,
                    const int symmetry, const bool read_cache,
                    const bool write_cache) {
  Netresult result;
  //const size_t boardsize = state->board.get_boardsize();
  //const size_t intersections = boardsize * boardsize;

  if (read_cache) {
    if (probe_cache(state, result)) {
      return result;
    }
  }
  if (ensemble == NONE) {
    assert(symmetry == -1);
    result = get_output_internal(state, IDENTITY_SYMMETRY);
  } else if (ensemble == DIRECT) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    result = get_output_internal(state, symmetry);
  } 
/*
  else if (ensemble == AVERAGE) {
    assert(symmetry == -1);
    for (int sym = 0; sym < NUM_SYMMETRIES; ++sym) {
      auto tmpresult = get_output_internal(state, sym);
      result.winrate[0] +=
          tmpresult.winrate[0] / static_cast<float>(NUM_SYMMETRIES);
      result.policy_pass +=
          tmpresult.policy_pass / static_cast<float>(NUM_SYMMETRIES);

      for (auto idx = size_t{0}; idx < intersections; idx++) {
        result.policy[idx] +=
            tmpresult.policy[idx] / static_cast<float>(NUM_SYMMETRIES);
      }
    }
  } 
*/
  else {
    assert(ensemble == RANDOM_SYMMETRY);
    assert(symmetry == -1);
    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    const auto rand_sym = rng.randfix<NUM_SYMMETRIES>();
    result = get_output_internal(state, rand_sym);
  }


  if (write_cache) {
    m_nncache.insert(state->board.get_hash(), result);
  }
  return result;
}

void Network::release_nn() {
  m_forward.release();
}

void Network::clear_cache() {
  m_nncache.clear();
}
