#ifndef CUDALAYER_H_INCLUDE
#define CUDALAYER_H_INCLUDE
#ifdef USE_CUDA
#include "cuda/CUDACommon.h"
#include "Winograd_helper.h"
#include <vector>

class CudaBatchnorm {
public:
  CudaBatchnorm() = default;
  CudaBatchnorm(int batch, int channels);
  ~CudaBatchnorm();
  // Forward: 須額外申請 device 端記憶體
  void Forward(const size_t batch, float *data,
               const float *const eltwise = nullptr);

  // cpu_Forward: 可以直從 host 端進行運算，無須額外申請 device 端記憶體
  void cpu_Forward(const int batch, std::vector<float> &data, float* eltwise = nullptr);

  void LoadingWeight(const std::vector<float> &means,
                     const std::vector<float> &stddevs);

private:
  static constexpr int spatial_size = NUM_INTERSECTIONS;
  int m_channels;
  int m_batch;

  bool is_loaded{false};
  float *cuda_means;
  float *cuda_stddevs;
  
};

class CudaConvolve {
public:
  CudaConvolve() = default;
  CudaConvolve(const size_t batch, const size_t filter,
               const size_t in_channels, const size_t out_channels);
  ~CudaConvolve();
  // Forward: 須額外申請 device 端記憶體
  void Forward(const int batch, float *input, float *output,
               void * cuda_scratch, size_t scratch_size, CudaHandel * handel);
  
  // cpu_Forward: 可以直從 host 端進行運算，無須額外申請 device 端記憶體
  // 不支持 cuDNN
  void cpu_Forward(const int batch, const std::vector<float> &input, std::vector<float> &output);
  void LoadingWeight(const std::vector<float> &weights, size_t & scratch_size);

private:
  static constexpr int width = CONV2D_SIZE;
  static constexpr int height = CONV2D_SIZE;
  static constexpr int spatial_size = width * height;
  int m_filter_dim;
  int m_batch;
  int m_filter;
  int m_in_channels;
  int m_out_channels;

#ifdef USE_CUDNN
  cudnnFilterDescriptor_t filter_desc;
  cudnnConvolutionDescriptor_t conv_desc;
  cudnnTensorDescriptor_t in_tensor_desc;
  cudnnTensorDescriptor_t out_tensor_desc;

  cudnnConvolutionFwdAlgo_t conv_algo;
  bool cudnn_applied{false};
#endif

  bool is_loaded{false};
  float *cuda_weights;
  float *cuda_col;
};

class CudaFullyConnect {
public:
  CudaFullyConnect() = default;
  CudaFullyConnect(const size_t batch, const size_t inputs, 
                   const size_t outputs, bool is_relu);
  ~CudaFullyConnect();
  // Forward: 須額外申請 device 端記憶體
  void Forward(const int batch, float *input,
               float *output, CudaHandel * handel);
  
  // cpu_Forward: 可以直從 host 端進行運算，無須額外申請 device 端記憶體 
  void cpu_Forward(const int batch, const std::vector<float> &input, std::vector<float> &output);

  void LoadingWeight(const std::vector<float> &weights,
                     const std::vector<float> &biases);
private:
  bool m_is_relu;
  int m_batch;
  int m_inputs;
  int m_outputs;

  bool is_loaded{false};
  float* cuda_weights;
  float* cuda_biases;
};



// TODO: 完成 Winograd Convolve
class CudaWinogradConvolve3 {
public:
  CudaWinogradConvolve3() = default;
  CudaWinogradConvolve3(const size_t batch,
                        const size_t in_channels, const size_t out_channels);
  ~CudaWinogradConvolve3();
  void LoadingWeight(const std::vector<float> & weights);
private:
  static constexpr size_t inflate_radio = WINOGRAD_TILE;

  int m_batch;
  int m_in_channels;
  int m_out_channels;

  bool is_loaded{false};
  float* cuda_weights;
};
#endif
#endif
