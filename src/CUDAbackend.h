#ifndef CUDABACKEND_H_INCLUDE
#define CUDABACKEND_H_INCLUDE

#include "NetPipe.h"

#ifdef USE_CUDA
#include "cuda/CUDALayers.h"
#include <vector>


class CUDAbackend : public ForwardPipe {
public:
  CUDAbackend();
  ~CUDAbackend();

  virtual void initialize(const int channels, int residual_blocks,
                          std::shared_ptr<ForwardPipeWeights> weights);

  virtual void forward(const std::vector<float> &input,
                       std::vector<float> &output_pol,
                       std::vector<float> &output_val);
  virtual void push_weights(unsigned int filter_size, unsigned int channels,
                            unsigned int outputs,
                            std::shared_ptr<const ForwardPipeWeights> weights);

private:
  float *cuda_input;
  float *cuda_output_pol;
  float *cuda_output_val;
  float *cuda_conv_out;
  float *cuda_conv_in;
  float *cuda_res;

  int m_input_channels;
  int m_residual_blocks;

  std::vector<CudaBatchnorm> batch_layer;
  std::vector<CudaConvolve> conv_layer;
  std::vector<CudaConvolve> conv_head;

  bool is_loaded;

};
#endif
#endif
