#ifndef CUDALAYER_H_INCLUDE
#define CUDALAYER_H_INCLUDE
#ifdef USE_CUDA
#include "NetPipe.h"
#include <vector>

class CudaBatchnorm {
public:
  CudaBatchnorm(int N, int channels);
  ~CudaBatchnorm();

  void Forward(const int batch, const size_t channels, float *data,
               const float *const eltwise = nullptr);

  void LoadingWeight(const std::vector<float> &means,
                     const std::vector<float> &stddevs);

private:
  static constexpr int spatial_size = NUM_INTERSECTIONS;
  int channels;
  int N;
  float *cuda_means;
  float *cuda_stddevs;
};

class CudaConvolve {
public:
  CudaConvolve(const size_t batch, const size_t filter, const size_t channels);
  ~CudaConvolve();
  void Forward(const int batch, float *input, float *output);
  void LoadingWeight(const std::vector<float> &weights,
                     const std::vector<float> &biases);

private:
  void im2col(const int batch, float *input, float *output);

  int N;
  int filter;
  int channels;
  float *cuda_weights;
  float *cuda_biases;
};

class cuda_test {
public:
  static void test_im2col(const size_t filter_size, const int channels,
                          const std::vector<float> &input,
                          std::vector<float> &output);

  static void test_batchnorm(const size_t channels, std::vector<float> &data,
                             const std::vector<float> &means,
                             const std::vector<float> &stddevs,
                             const float *const eltwise = nullptr);

  static void test_gemm_gpu(bool TA, bool TB, int M, int N, int K, float ALPHA,
                            const float *A, int lda, const float *B, int ldb,
                            float BETA, float *C, int ldc);
};
#endif
#endif
