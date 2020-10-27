
#include "Network.h"

#ifdef USE_ZLIB
#include "zlib.h"
#endif

#include "CPUBackend.h"
#include "Board.h"
#include "GameState.h"
#include "Random.h"
#include "Utils.h"
#include "Blas.h"
#include "config.h"

#ifdef USE_CUDA
#include "CUDABackend.h"
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

Network::~Network() {
    m_forward->destroy();  
}

void Network::initialize(const int playouts, const std::string &weightsfile) {

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
    set_playouts(playouts);

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

void Network::set_playouts(const int playouts) {
    const size_t cache_size = option<int>("cache_moves") * playouts;
    m_cache.resize(cache_size);
}

// TODO: probe_cache 翻轉搜尋?
bool Network::probe_cache(const GameState *const state,
                          Network::Netresult &result,
                          const int symmetry) {

    const bool success = m_cache.lookup(state->board.get_hash(), result);

    if (success) {
        // result = Model::get_result_form_cache(result);
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
      if (m_cache.lookup(hash, result)) {
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

    return success;
}

void Network::dummy_forward(std::vector<float> &policy,
                            std::vector<float> &ownership,
                            std::vector<float> &final_score,
                            std::vector<float> &values) {

    auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
    auto dis = std::uniform_real_distribution<float>(0.0, 1.0);
    for (auto & p : policy) {
        p = dis(rng);
    }
    const auto acc = std::accumulate(std::begin(policy),
                                         std::end(policy), 0.0f);
    for (auto & p : policy) {
        p /= acc;
    }

    values[0] = 0.0f;
    values[1] = 1.0f;
    // values[2] = 0.0f;

    for (auto &owner : ownership) {
        owner = 0.0f;
    }

    final_score[0] = 0.0f;
}

Network::Netresult Network::get_output_internal(const GameState *const state,
                                                const int symmetry) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);

    auto policy_out = std::vector<float>(POTENTIAL_MOVES);
    auto scorebelief_out = std::vector<float>(OUTPUTS_SCOREBELIEF * NUM_INTERSECTIONS);
    auto finalscore_out = std::vector<float>(FINAL_SCORE);
    auto ownership_out = std::vector<float>(OUTPUTS_OWNERSHIP * NUM_INTERSECTIONS);
    auto winrate_out = std::vector<float>(VALUE_MISC);

    const auto boardsize = state->board.get_boardsize();
    const auto input_planes = Model::gather_planes(state, symmetry);
    const auto input_features = Model::gather_features(state);
    if (m_forward->valid()) {
        m_forward->forward(boardsize, input_planes, input_features,
                           policy_out, scorebelief_out, ownership_out, finalscore_out, winrate_out);
    } else {
        dummy_forward(policy_out, ownership_out, finalscore_out, winrate_out);
    }

    const auto result = Model::get_result(state,
                                          policy_out,
                                          scorebelief_out,
                                          ownership_out,
                                          finalscore_out,
                                          winrate_out,
                                          option<float>("softmax_temp"), symmetry);

    return result;
}

Network::Netresult
Network::get_output(const GameState *const state,
                    const Ensemble ensemble,
                    const int symmetry,
                    const bool read_cache,
                    const bool write_cache) {

    Netresult result;

    if (read_cache) {
        if (probe_cache(state, result, symmetry)) {
            return result;
        }
    }
    if (ensemble == NONE) {
        assert(symmetry == -1);
        result = get_output_internal(state, IDENTITY_SYMMETRY);
    } else if (ensemble == DIRECT) {
        assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
        result = get_output_internal(state, symmetry);
    } else {
        assert(ensemble == RANDOM_SYMMETRY);
        assert(symmetry == -1);
        auto rng = Random<random_t::XoroShiro128Plus>::get_Rng();
        const auto rand_sym = rng.randfix<NUM_SYMMETRIES>();
        result = get_output_internal(state, rand_sym);
    }

    if (write_cache) {
        m_cache.insert(state->board.get_hash(), result);
    }
    return result;
}

void Network::release_nn() {
    m_forward->release();
}

void Network::clear_cache() {
    m_cache.clear();
}
