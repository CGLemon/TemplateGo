#include "config.h"

#include <algorithm>
#include <cassert>
#include <vector>

#include "LZ/LZCPUbackend.h"
#include "LZ/LZNetParameters.h"

#include "Utils.h"

using namespace Utils;


namespace LZ {

CPUbackend::CPUbackend() {
  auto_printf("Using cpu blas network.\n");
}

void CPUbackend::initialize(std::shared_ptr<NNWeights> weights) {
  m_residual_channels = weights->get_num_channels(0);
  m_residual_blocks = weights->get_num_residuals();
  m_weights = weights;
  LZModel::transform(true, m_weights);

}

void CPUbackend::forward(const std::vector<float> &input,
                         std::vector<float> &output_pol,
                         std::vector<float> &output_val) {


  using batchnorm = Batchnorm<conv_size>;
  using convolve_3 = winograd_convolve3<conv_size>;
  using convolve_1 = Convolve1<conv_size>;

  size_t output_channels = m_residual_channels;
  // 因為第一層進入的 channel 可能會比 resnet 的 channel 還多
  // 反之也可能 ，所以要選擇最大的
  size_t input_channels = std::max(static_cast<size_t>(output_channels),
                                   static_cast<size_t>(LZ::INPUT_CHANNELS));

  const auto workspace_size = 
                 convolve_3::get_workspace_size(input_channels, output_channels);
  const auto winograd_V_size = workspace_size.first;
  const auto winograd_M_size = workspace_size.second;


  auto winograd_V = std::vector<float>(winograd_V_size);
  auto winograd_M = std::vector<float>(winograd_M_size);


  auto conv_out = std::vector<float>(output_channels * NUM_INTERSECTIONS);
  auto conv_in = std::vector<float>(output_channels * NUM_INTERSECTIONS);
  auto res = std::vector<float>(output_channels * NUM_INTERSECTIONS);


  auto policy_conv = std::vector<float>(LZ::OUTPUTS_POLICY * NUM_INTERSECTIONS);
  auto value_conv = std::vector<float>(LZ::OUTPUTS_VALUE * NUM_INTERSECTIONS);
  auto value_layer = std::vector<float>(LZ::VALUE_LAYER);


  input_channels = LZ::INPUT_CHANNELS;

  convolve_3::Forward(input_channels,
                      output_channels, input, 
                      m_weights->m_ip_conv.m_conv.m_weights, 
                      winograd_V, winograd_M,conv_out);
  batchnorm::Forward(output_channels, conv_out,
                     m_weights->m_ip_conv.m_batchnorm.m_means,
                     m_weights->m_ip_conv.m_batchnorm.m_stddevs);

  
  input_channels = m_residual_channels;
  for (auto i = size_t{0}; i < m_residual_blocks; i++) {
    std::swap(conv_out, conv_in);
    convolve_3::Forward(input_channels, output_channels, conv_in,
                        m_weights->m_res_blocks[i].m_conv_blocks[0].m_conv.m_weights, 
                        winograd_V, winograd_M, conv_out);
    batchnorm::Forward(output_channels, conv_out,
                       m_weights->m_res_blocks[i].m_conv_blocks[0].m_batchnorm.m_means,
                       m_weights->m_res_blocks[i].m_conv_blocks[0].m_batchnorm.m_stddevs);

    std::swap(conv_in, res);
    std::swap(conv_out, conv_in);
    convolve_3::Forward(input_channels, output_channels, conv_in,
                        m_weights->m_res_blocks[i].m_conv_blocks[1].m_conv.m_weights,
                        winograd_V, winograd_M, conv_out);
    batchnorm::Forward(output_channels, conv_out,
                       m_weights->m_res_blocks[i].m_conv_blocks[1].m_batchnorm.m_means,
                       m_weights->m_res_blocks[i].m_conv_blocks[1].m_batchnorm.m_stddevs, res.data());

  }


  input_channels = output_channels;
  // policy head
  convolve_1::Forward(input_channels, LZ::OUTPUTS_POLICY, conv_out, 
                      m_weights->m_conv_pol.m_conv.m_weights,
                      policy_conv);

  batchnorm::Forward(LZ::OUTPUTS_POLICY, policy_conv, 
                     m_weights->m_conv_pol.m_batchnorm.m_means,
                     m_weights->m_conv_pol.m_batchnorm.m_stddevs);

  FullyConnect::Forward(LZ::OUTPUTS_POLICY * NUM_INTERSECTIONS, LZ::POTENTIAL_MOVES,
                        policy_conv, 
                        m_weights->m_fc_pol.m_weights,
                        m_weights->m_fc_pol.m_biases, 
                        output_pol, false);

  // value head
  convolve_1::Forward(input_channels, LZ::OUTPUTS_VALUE, conv_out, 
                      m_weights->m_conv_val.m_conv.m_weights,
                      value_conv);

  batchnorm::Forward(LZ::OUTPUTS_VALUE, value_conv, 
                     m_weights->m_conv_val.m_batchnorm.m_means,
                     m_weights->m_conv_val.m_batchnorm.m_stddevs);

  FullyConnect::Forward(LZ::OUTPUTS_VALUE * NUM_INTERSECTIONS, LZ::VALUE_LAYER,
                        value_conv, 
                        m_weights->m_fc1_val.m_weights,
                        m_weights->m_fc1_val.m_biases, 
                        value_layer, true);

  FullyConnect::Forward(LZ::VALUE_LAYER, LZ::VALUE_LABELS,
                        value_layer, 
                        m_weights->m_fc2_val.m_weights,
                        m_weights->m_fc2_val.m_biases, 
                        output_val, false);
}
}
