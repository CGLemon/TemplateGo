#ifndef LZCUDABACKEND_H_INCLUDE
#define LZCUDABACKEND_H_INCLUDE


#ifdef USE_CUDA
#include "cuda/CUDACommon.h"
#include "cuda/CUDAKernels.h"
#include "cuda/CUDALayers.h"
#include "LZ/LZNetParameters.h"
#include "LZ/LZModel.h"

#include <vector>
#include <array>
#include <mutex>

namespace LZ {
class CUDAbackend : public NNpipe {
public:
  CUDAbackend();
  ~CUDAbackend();
  using NNWeights = LZModel::ForwardPipeWeights;

  virtual void initialize(std::shared_ptr<NNWeights> weights);

  virtual void forward(const std::vector<float> &input,
                       std::vector<float> &output_pol,
                       std::vector<float> &output_val);

private:
  void pushing_weights(std::shared_ptr<NNWeights> weights);


  std::shared_ptr<NNWeights> m_weights;

  
  int m_residual_channels;
  int m_residual_blocks;

  
  float *cuda_input;
  float *cuda_output_pol;
  float *cuda_output_val;

  std::array<float*, 3> cuda_conv_temp;

  std::array<float*, 1> cuda_pol_layer;
  std::array<float*, 2> cuda_val_layer;
  
  std::vector<CudaConvolve> conv_layer;
  std::vector<CudaBatchnorm> bnorm_layer;

  CudaConvolve poliy_conv;
  CudaBatchnorm poliy_bnorm;
  CudaFullyConnect poliy_fc;
  
  CudaConvolve value_conv;
  CudaBatchnorm value_bnorm;
  CudaFullyConnect value_fc1;
  CudaFullyConnect value_fc2;

  bool is_applied;
  std::mutex m_mtx;
};
}

#endif
#endif
