#ifndef LZCPUBACKEND_H_INCLUDE
#define LZCPUBACKEND_H_INCLUDE

#include "LZ/LZModel.h"
#include <vector>
#include <array>

namespace LZ {
class CPUbackend : public NNpipe {
public:
  CPUbackend();
  using NNWeights = LZModel::ForwardPipeWeights;

  virtual void initialize(std::shared_ptr<NNWeights> weights);

  virtual void forward(const std::vector<float> &input,
                       std::vector<float> &output_pol,
                       std::vector<float> &output_val);

private:
  std::shared_ptr<NNWeights> m_weights;
  int m_residual_channels;
  int m_residual_blocks;


  std::vector<float> m_conv_in;
  std::vector<float> m_conv_out;
  std::vector<float> m_res;

  std::vector<float> m_winograd_V;
  std::vector<float> m_winograd_M;


  std::vector<float> m_policy_conv;

  std::vector<float> m_value_conv;
  std::vector<float> m_value_layer;
};

}


#endif
