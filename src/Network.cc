/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Gian-Carlo Pascutto and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#include "Network.h"

#ifdef USE_ZLIB
#include "zlib.h"
#endif

#include "Board.h"
#include "CPUbackend.h"
#include "GameState.h"
#include "NetPipe.h"
#include "Random.h"
#include "Utils.h"
#include "blas/CPULayers.h"
#include "cfg.h"

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
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
using namespace Utils;

bool check_net_weight(std::vector<float> &weights, int count) {
  bool succeess = (weights.size() == count);
  if (succeess) {
    weights.reserve(count);
  } else {
    auto_printf("Network file struct is wrong.\n");
  }
  return succeess;
}

bool Network::check_net_format(Networkfile_t file_type,
                               const int resnet_channels,
                               const int resnet_blocks) {

  if (file_type == Networkfile_t::LEELAZ) {
    if (OUTPUTS_POLICY != 2 && OUTPUTS_VALUE != 1 && INPUT_CHANNELS != 18 &&
        VALUE_LAYER != 256 && VALUE_LABELS != 1) {

      auto_printf("Network default struct is wrong.\n");
      return false;
    } else {

      if (!check_net_weight(m_fwd_weights->m_conv_weights[0],
                            INPUT_CHANNELS * 9 * resnet_channels))
        return false;
      if (!check_net_weight(m_fwd_weights->m_conv_biases[0], resnet_channels))
        return false;
      if (!check_net_weight(m_fwd_weights->m_batchnorm_means[0],
                            resnet_channels))
        return false;
      if (!check_net_weight(m_fwd_weights->m_batchnorm_stddevs[0],
                            resnet_channels))
        return false;

      for (int i = 0; i < resnet_blocks; ++i) {
        const int id = i + 1;
        if (!check_net_weight(m_fwd_weights->m_conv_weights[id],
                              resnet_channels * 9 * resnet_channels))
          return false;
        if (!check_net_weight(m_fwd_weights->m_conv_biases[id],
                              resnet_channels))
          return false;
        if (!check_net_weight(m_fwd_weights->m_batchnorm_means[id],
                              resnet_channels))
          return false;
        if (!check_net_weight(m_fwd_weights->m_batchnorm_stddevs[id],
                              resnet_channels))
          return false;
      }

      if (!check_net_weight(m_fwd_weights->m_conv_pol_w,
                            OUTPUTS_POLICY * 1 * resnet_channels))
        return false;
      if (!check_net_weight(m_fwd_weights->m_conv_pol_b, OUTPUTS_POLICY))
        return false;
      if (!check_net_weight(m_bn_pol_w1, OUTPUTS_POLICY))
        return false;
      if (!check_net_weight(m_bn_pol_w2, OUTPUTS_POLICY))
        return false;
      if (!check_net_weight(m_ip_pol_w, OUTPUTS_POLICY * NUM_INTERSECTIONS *
                                            POTENTIAL_MOVES))
        return false;
      if (!check_net_weight(m_ip_pol_b, POTENTIAL_MOVES))
        return false;

      if (!check_net_weight(m_fwd_weights->m_conv_val_w,
                            OUTPUTS_VALUE * 1 * resnet_channels))
        return false;
      if (!check_net_weight(m_fwd_weights->m_conv_val_b, OUTPUTS_VALUE))
        return false;
      if (!check_net_weight(m_bn_val_w1, OUTPUTS_VALUE))
        return false;
      if (!check_net_weight(m_bn_val_w2, OUTPUTS_VALUE))
        return false;
      if (!check_net_weight(m_ip1_val_w,
                            OUTPUTS_VALUE * NUM_INTERSECTIONS * VALUE_LAYER))
        return false;
      if (!check_net_weight(m_ip1_val_b, VALUE_LAYER))
        return false;
      if (!check_net_weight(m_ip2_val_w, VALUE_LAYER * VALUE_LABELS))
        return false;
      if (!check_net_weight(m_ip2_val_b, VALUE_LABELS))
        return false;

      return true;
    }
  }
  auto_printf("Weights file is the wrong type.\n");
  return false;
}

template <class container> void process_bn_var(container &weights) {
  constexpr float epsilon = 1e-5f;
  for (auto &&w : weights) {
    w = 1.0f / std::sqrt(w + epsilon);
  }
}

std::pair<int, int> Network::load_leelaz_network(std::istream &wtfile) {

  auto_printf("Detecting residual layers...");

  std::vector<float> weights;
  int linecount = 1;
  int channels = 0;
  auto line = std::string{};
  while (std::getline(wtfile, line)) {
    auto iss = std::stringstream{line};
    if (linecount == 2) {
      auto count = std::distance(std::istream_iterator<std::string>(iss),
                                 std::istream_iterator<std::string>());
      auto_printf("%d channels...", count);
      channels = count;
    }
    linecount++;
  }

  const int version_lines = 1;
  const int tower_head_conv_lines = 4;
  const int linear_lines = 14;
  const int not_residual_lines =
      version_lines + tower_head_conv_lines + linear_lines;
  const int residual_lines = linecount - not_residual_lines;
  if (residual_lines % 8 != 0) {
    auto_printf("\nInconsistent number of weights in the file.\n");
    return {0, 0};
  }
  const int residual_blocks = residual_lines / 8;
  auto_printf("%d blocks.\n", residual_blocks);

  wtfile.clear();
  wtfile.seekg(0, std::ios::beg);

  std::getline(wtfile, line);
  const int plain_conv_layers = 1 + (residual_blocks * 2);
  const int plain_conv_wts = plain_conv_layers * 4;
  linecount = 0;
  
  while (std::getline(wtfile, line)) {
    std::vector<float> weights;
    float weight;
    std::istringstream iss(line);
    while (iss >> weight) {
      weights.emplace_back(weight);
    }
    if (linecount < plain_conv_wts) {
      if (linecount % 4 == 0) {
        m_fwd_weights->m_conv_weights.emplace_back(weights);
      } else if (linecount % 4 == 1) {
        m_fwd_weights->m_conv_biases.emplace_back(weights);
      } else if (linecount % 4 == 2) {
        m_fwd_weights->m_batchnorm_means.emplace_back(weights);
      } else if (linecount % 4 == 3) {
        process_bn_var(weights);
        m_fwd_weights->m_batchnorm_stddevs.emplace_back(weights);
      }
    } else {
      switch (linecount - plain_conv_wts) {
      case 0:
        m_fwd_weights->m_conv_pol_w = std::move(weights);
        break;
      case 1:
        m_fwd_weights->m_conv_pol_b = std::move(weights);
        break;
      case 2:
        m_bn_pol_w1 = std::move(weights);
        break;
      case 3:
        m_bn_pol_w2 = std::move(weights);
        break;
      case 4:
        m_ip_pol_w = std::move(weights);
        break;
      case 5:
        m_ip_pol_b = std::move(weights);
        break;
      case 6:
        m_fwd_weights->m_conv_val_w = std::move(weights);
        break;
      case 7:
        m_fwd_weights->m_conv_val_b = std::move(weights);
        break;
      case 8:
        m_bn_val_w1 = std::move(weights);
        break;
      case 9:
        m_bn_val_w2 = std::move(weights);
        break;
      case 10:
        m_ip1_val_w = std::move(weights);
        break;
      case 11:
        m_ip1_val_b = std::move(weights);
        break;
      case 12:
        m_ip2_val_w = std::move(weights);
        break;
      case 13:
        m_ip2_val_b = std::move(weights);
        break;
      }
    }
    linecount++;
  }
  process_bn_var(m_bn_pol_w2);
  process_bn_var(m_bn_val_w2);
  
  if (!check_net_format(Networkfile_t::LEELAZ, channels, residual_blocks)) {
    m_fwd_weights.reset();
    return {0, 0};
  }

  return {channels, residual_blocks};
}

std::pair<int, int> Network::load_network_file(const std::string &filename,
                                               Networkfile_t file_type) {

  auto buffer = std::stringstream{};

#ifdef USE_ZLIB
  auto gzhandle = gzopen(filename.c_str(), "rb");
  if (gzhandle == nullptr) {
    auto_printf("Could not open weights file: %s\n", filename.c_str());
    return {0, 0};
  }

  constexpr auto chunkBufferSize = 64 * 1024;
  std::vector<char> chunkBuffer(chunkBufferSize);

  while (true) {
    auto bytesRead = gzread(gzhandle, chunkBuffer.data(), chunkBufferSize);
    if (bytesRead == 0)
      break;
    if (bytesRead < 0) {
      auto_printf("Failed to decompress or read: %s\n", filename.c_str());
      gzclose(gzhandle);
      return {0, 0};
    }
    assert(bytesRead <= chunkBufferSize);
    buffer.write(chunkBuffer.data(), bytesRead);
  }
  gzclose(gzhandle);
#else

  std::ifstream weights_file(filename.c_str());
  auto stream_line = std::string{};
  while (std::getline(weights_file, stream_line)) {
    buffer << stream_line << std::endl;
  }
  weights_file.close();

#endif

  auto line = std::string{};
  if (file_type == Networkfile_t::LEELAZ) {
    int format_version = -1;
    if (std::getline(buffer, line)) {
      auto iss = std::stringstream{line};
      iss >> format_version;
      if (iss.fail() || (format_version != 1 && format_version != 2)) {
        auto_printf("Weights file is the wrong version.\n");
        return {0, 0};
      } else {
        if (format_version == 2) {
          m_value_head_not_stm = true;
          auto_printf("Loading ELF OpenGO network file.\n");
        } else {
          m_value_head_not_stm = false;
          auto_printf("Loading Leelaz network file.\n");
        }
        return load_leelaz_network(buffer);
      }
    }
  }
  return {0, 0};
}

void Network::init_leelaz_batchnorm_weights() {
  auto bias_size = m_fwd_weights->m_conv_biases.size();
  for (auto i = size_t{0}; i < bias_size; i++) {
    auto means_size = m_fwd_weights->m_batchnorm_means[i].size();
    for (auto j = size_t{0}; j < means_size; j++) {
      m_fwd_weights->m_batchnorm_means[i][j] -=
          m_fwd_weights->m_conv_biases[i][j];
      m_fwd_weights->m_conv_biases[i][j] = 0.0f;
    }
  }
  for (auto i = size_t{0}; i < m_bn_val_w1.size(); i++) {
    m_bn_val_w1[i] -= m_fwd_weights->m_conv_val_b[i];
    m_fwd_weights->m_conv_val_b[i] = 0.0f;
  }

  for (auto i = size_t{0}; i < m_bn_pol_w1.size(); i++) {
    m_bn_pol_w1[i] -= m_fwd_weights->m_conv_pol_b[i];
    m_fwd_weights->m_conv_pol_b[i] = 0.0f;
  }
}

std::unique_ptr<ForwardPipe> &&
Network::init_net(int channels, int residual_blocks,
                  std::unique_ptr<ForwardPipe> &&pipe) {
  
  pipe->initialize(channels, residual_blocks, m_fwd_weights);
  pipe->push_weights(WINOGRAD_ALPHA, INPUT_CHANNELS, channels, m_fwd_weights);

  return std::move(pipe);
}

void Network::initialize(int playouts, const std::string &weightsfile,
                         Networkfile_t file_type) {

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
  m_fwd_weights = std::make_shared<ForwardPipeWeights>();
  m_nncache.resize(cfg_cache_moves * playouts);

  size_t channels, residual_blocks;
  std::tie(channels, residual_blocks) =
      load_network_file(weightsfile, file_type);
  if (channels == 0) {
    exit(EXIT_FAILURE);
  }

  init_leelaz_batchnorm_weights();

#ifdef USE_CUDA
  using backend = CUDAbackend;
#else
  using backend = CPUbackend;
#endif 

  m_forward =
      init_net(channels, residual_blocks, std::make_unique<backend>());

  m_fwd_weights.reset();
  auto_printf("Pushing Network is complete.\n");
}

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

void Network::fill_input_plane_pair(const std::shared_ptr<Board> board,
                                    std::vector<float>::iterator black,
                                    std::vector<float>::iterator white,
                                    const int symmetry) {
  for (int idx = 0; idx < NUM_INTERSECTIONS; idx++) {
    const int sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
    const auto sym_pair = get_intersections_pair(sym_idx, BOARD_SIZE);
    const int x = sym_pair.first;
    const int y = sym_pair.second;
    const int color = board->get_state(x, y);
    if (color == Board::BLACK) {
      black[idx] = static_cast<float>(true);
    } else if (color == Board::WHITE) {
      white[idx] = static_cast<float>(true);
    }
  }
}

std::vector<float> Network::gather_features(const GameState *const state,
                                            const int symmetry) {
  assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
  auto input_data = std::vector<float>(INPUT_CHANNELS * NUM_INTERSECTIONS, 0.0f);

  const auto to_move = state->board.get_to_move();
  const auto blacks_move = to_move == Board::BLACK;

  const auto black_it =
      blacks_move ? begin(input_data)
                  : begin(input_data) + INPUT_MOVES * NUM_INTERSECTIONS;
  const auto white_it =
      blacks_move ? begin(input_data) + INPUT_MOVES * NUM_INTERSECTIONS
                  : begin(input_data);
  const auto to_move_it =
      blacks_move
          ? begin(input_data) + 2 * INPUT_MOVES * NUM_INTERSECTIONS
          : begin(input_data) + (2 * INPUT_MOVES + 1) * NUM_INTERSECTIONS;

  const auto moves =
      std::min<size_t>(state->board.get_movenum() + 1, INPUT_MOVES);
  // Go back in time, fill history boards

  for (auto h = size_t{0}; h < moves; h++) {
    // collect white, black occupation planes
    fill_input_plane_pair(state->get_past_board(h),
                          black_it + h * NUM_INTERSECTIONS,
                          white_it + h * NUM_INTERSECTIONS, symmetry);
  }
  std::fill(to_move_it, to_move_it + NUM_INTERSECTIONS, static_cast<float>(true));

  return input_data;
}

Network::Netresult Network::get_output_internal(const GameState *const state,
                                                const int symmetry) {
  assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
  constexpr int width = BOARD_SIZE;
  constexpr int height = BOARD_SIZE;
  const auto input_data = gather_features(state, symmetry);
  std::vector<float> policy_data(OUTPUTS_POLICY * width * height);
  std::vector<float> value_data(OUTPUTS_VALUE * width * height);

  m_forward->forward(input_data, policy_data, value_data);

  // Get the moves
  Batchnorm::Forward(OUTPUTS_POLICY, policy_data, m_bn_pol_w1.data(),
                     m_bn_pol_w2.data());

  std::vector<float> policy_out(POTENTIAL_MOVES);
  std::vector<float> winrate_data(VALUE_LAYER);
  std::vector<float> winrate_out(VALUE_LABELS);

  FullyConnect::Forward(OUTPUTS_POLICY * NUM_INTERSECTIONS, POTENTIAL_MOVES,
                        policy_data, m_ip_pol_w, m_ip_pol_b, policy_out, false);
  const auto outputs = Activation::softmax(policy_out, cfg_softmax_temp);

  // Now get the value
  Batchnorm::Forward(OUTPUTS_VALUE, value_data, m_bn_val_w1.data(),
                     m_bn_val_w2.data());
  FullyConnect::Forward(OUTPUTS_VALUE * NUM_INTERSECTIONS, VALUE_LAYER,
                        value_data, m_ip1_val_w, m_ip1_val_b, winrate_data,
                        true);

  FullyConnect::Forward(VALUE_LAYER, VALUE_LABELS, winrate_data, m_ip2_val_w,
                        m_ip2_val_b, winrate_out, false);

  // Map TanH output range [-1..1] to [0..1] range
  const auto winrate = (1.0f + std::tanh(winrate_out[0])) / 2.0f;

  Netresult result;

  for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
    const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
    result.policy[sym_idx] = outputs[idx];
  }

  result.policy_pass = outputs[NUM_INTERSECTIONS];
  result.winrate[0] = winrate;

  return result;
}

Network::Netresult
Network::get_output(const GameState *const state, const Ensemble ensemble,
                    const int symmetry, const bool read_cache,
                    const bool write_cache, const bool force_selfcheck) {
  Netresult result;

  if (state->board.get_boardsize() != BOARD_SIZE) {
    return result;
  }

  if (read_cache) {
    if (probe_cache(state, result)) {
      return result;
    }
  }

  if (ensemble == DIRECT) {
    assert(symmetry >= 0 && symmetry < NUM_SYMMETRIES);
    result = get_output_internal(state, symmetry);
  } else if (ensemble == AVERAGE) {
    assert(symmetry == -1);
    for (int sym = 0; sym < NUM_SYMMETRIES; ++sym) {
      auto tmpresult = get_output_internal(state, sym);
      result.winrate[0] +=
          tmpresult.winrate[0] / static_cast<float>(NUM_SYMMETRIES);
      result.policy_pass +=
          tmpresult.policy_pass / static_cast<float>(NUM_SYMMETRIES);

      for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
        result.policy[idx] +=
            tmpresult.policy[idx] / static_cast<float>(NUM_SYMMETRIES);
      }
    }
  } else {
    assert(ensemble == RANDOM_SYMMETRY);
    assert(symmetry == -1);
    const auto rand_sym = Random::get_Rng().randfix<NUM_SYMMETRIES>();
    result = get_output_internal(state, rand_sym);
  }

  // v2 format (ELF Open Go) returns black value, not stm
  if (m_value_head_not_stm) {
    if (state->board.get_to_move() == Board::WHITE) {
      result.winrate[0] = 1.0f - result.winrate[0];
    }
  }

  if (write_cache) {
    m_nncache.insert(state->board.get_hash(), result);
  }
  return result;
}
