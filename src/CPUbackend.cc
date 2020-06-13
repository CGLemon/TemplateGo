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

#include "config.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include "blas/CPULayers.h"

#include "CPUbackend.h"
#include "NetPipe.h"

#include "Utils.h"

using namespace Utils;

CPUbackend::CPUbackend() {
  auto_printf("Using cpu blas network.\n");
}


void CPUbackend::initialize(int channels, int residual_blocks,
                            std::shared_ptr<ForwardPipeWeights> weights) {
  m_input_channels = channels;
  m_residual_blocks = residual_blocks;

  auto weight_index = size_t{0};

  weights->m_conv_weights[weight_index] = winograd_transform_f(
      weights->m_conv_weights[weight_index], m_input_channels, INPUT_CHANNELS);
  weight_index++;

  for (auto i = size_t{0}; i < m_residual_blocks * 2; i++) {
    weights->m_conv_weights[weight_index] =
        winograd_transform_f(weights->m_conv_weights[weight_index],
                             m_input_channels, m_input_channels);
    weight_index++;
  }
}

void CPUbackend::forward(const std::vector<float> &input,
                         std::vector<float> &output_pol,
                         std::vector<float> &output_val) {
  // Input convolution
  constexpr auto P = WINOGRAD_P;
  // Calculate output channels
  const auto output_channels = m_input_channels;
  // input_channels is the maximum number of input channels of any
  // convolution. Residual blocks are identical, but the first convolution
  // might be bigger when the network has very few filters
  const auto input_channels = std::max(static_cast<size_t>(output_channels),
                                       static_cast<size_t>(INPUT_CHANNELS));
  auto conv_out = std::vector<float>(output_channels * NUM_INTERSECTIONS);

  auto V = std::vector<float>(WINOGRAD_TILE * input_channels * P);
  auto M = std::vector<float>(WINOGRAD_TILE * output_channels * P);

  using batchnorm = Batchnorm;
  using Convolve_3 = winograd_convolve3;

  Convolve_3::Forward(output_channels, input, m_weights->m_conv_weights[0],
                      V, M, conv_out);
  batchnorm::Forward(output_channels, conv_out,
                     m_weights->m_batchnorm_means[0].data(),
                     m_weights->m_batchnorm_stddevs[0].data());
  

  // Residual tower
  auto conv_in = std::vector<float>(output_channels * NUM_INTERSECTIONS);
  auto res = std::vector<float>(output_channels * NUM_INTERSECTIONS);
  for (auto i = size_t{1}; i < m_weights->m_conv_weights.size(); i += 2) {
    auto output_channels = m_input_channels;
    std::swap(conv_out, conv_in);
    Convolve_3::Forward(output_channels, conv_in,
                        m_weights->m_conv_weights[i], V, M, conv_out);
    batchnorm::Forward(output_channels, conv_out,
                       m_weights->m_batchnorm_means[i].data(),
                       m_weights->m_batchnorm_stddevs[i].data());
    
    std::swap(conv_in, res);
    std::swap(conv_out, conv_in);
    Convolve_3::Forward(output_channels, conv_in,
                        m_weights->m_conv_weights[i + 1], V, M, conv_out);
    batchnorm::Forward(
        output_channels, conv_out, m_weights->m_batchnorm_means[i + 1].data(),
        m_weights->m_batchnorm_stddevs[i + 1].data(), res.data());
    
  }
  Convolve_1::Forward(OUTPUTS_POLICY, conv_out, m_conv_pol_w, m_conv_pol_b,
                      output_pol);
  Convolve_1::Forward(OUTPUTS_VALUE, conv_out, m_conv_val_w, m_conv_val_b,
                      output_val);
}


void CPUbackend::push_weights(
    unsigned int /*filter_size*/, unsigned int /*channels*/,
    unsigned int outputs, std::shared_ptr<const ForwardPipeWeights> weights) {

  m_weights = weights;

  // Output head convolutions
  m_conv_pol_w = weights->m_conv_pol_w;
  m_conv_pol_b.resize(m_conv_pol_w.size() / outputs, 0.0f);
  m_conv_val_w = weights->m_conv_val_w;
  m_conv_val_b.resize(m_conv_val_w.size() / outputs, 0.0f);
}
