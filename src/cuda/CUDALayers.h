#ifndef CUDALAYER_H_INCLUDE
#define CUDALAYER_H_INCLUDE
#ifdef USE_CUDA
#include "NetPipe.h"
#include <vector>

class CudaBatchnorm {
public:
  CudaBatchnorm(int batch, int channels);
  ~CudaBatchnorm();
  // Forward: 須額外申請 device 端記憶體
  void Forward(const int batch, float *data,
               const float *const eltwise = nullptr);

  // cpu_Forward: 可以直從 host 端進行運算，無須額外申請 device 端記憶體
  void cpu_Forward(const int batch, std::vector<float> &data, float* eltwise = nullptr);

  void LoadingWeight(const std::vector<float> &means,
                     const std::vector<float> &stddevs);

private:
  static constexpr int spatial_size = NUM_INTERSECTIONS;
  int channels;
  int batch;
  float *cuda_means;
  float *cuda_stddevs;
  bool is_loaded;
};

class CudaConvolve {
public:
  CudaConvolve(const size_t batch, const size_t filter, const size_t in_channels, const size_t out_channels);
  ~CudaConvolve();
  // Forward: 須額外申請 device 端記憶體
  void Forward(const int batch, float *input, float *output);
  
  // cpu_Forward: 可以直從 host 端進行運算，無須額外申請 device 端記憶體
  void cpu_Forward(const int batch, const std::vector<float> &input, std::vector<float> &output);
  void LoadingWeight(const std::vector<float> &weights);

private:
  static constexpr int width = CONV2D_SIZE;
  static constexpr int height = CONV2D_SIZE;
  static constexpr int spatial_size = width * height;
  int filter_dim;
  int batch;
  int filter;
  int in_channels;
  int out_channels;
  float *cuda_weights;
  float *cuda_col;

  size_t w_s;
  bool is_loaded;
};

class CudaFullyConnect {
public:
  CudaFullyConnect(const size_t batch, const size_t inputs, const size_t outputs);

private:
  int m_batch;
  int m_inputs;
  int m_outputs;
  bool is_loaded;
  float cuda_weights;
  float cuda_biases;

};

#endif
#endif
