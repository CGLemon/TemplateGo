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
};

}


#endif
