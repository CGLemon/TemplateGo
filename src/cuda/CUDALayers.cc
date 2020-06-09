#include "cuda/CUDALayers.h"
#include "cuda/CUDACommon.h"
#include "cuda/CUDAKernels.h"
#include <cassert>
#ifdef USE_CUDA
CudaBatchnorm::CudaBatchnorm(int MAX_N, int output_channels) {

  channels = output_channels;
  N = MAX_N;
}

CudaBatchnorm::~CudaBatchnorm() {
  ReportCUDAErrors(cudaFree(cuda_means));
  ReportCUDAErrors(cudaFree(cuda_stddevs));
}

void CudaBatchnorm::Forward(const int batch, const size_t channels, float *data,
                            const float *const eltwise) {
  assert(batch <= N);
  batchNorm(data, cuda_means, cuda_stddevs, batch, channels, spatial_size,
            eltwise);
}

void CudaBatchnorm::LoadingWeight(const std::vector<float> &means,
                                  const std::vector<float> &stddevs) {

  const size_t weights_size = sizeof(float) * channels;
  assert(weights_size == sizeof(float) * means.size() &&
         weights_size == sizeof(float) * stddevs.size());
  ReportCUDAErrors(cudaMalloc(&cuda_means, weights_size));
  ReportCUDAErrors(cudaMalloc(&cuda_stddevs, weights_size));

  ReportCUDAErrors(cudaMemcpy(cuda_means, means.data(), weights_size,
                              cudaMemcpyHostToDevice));
  ReportCUDAErrors(cudaMemcpy(cuda_stddevs, stddevs.data(), weights_size,
                              cudaMemcpyHostToDevice));
}

CudaConvolve::CudaConvolve(const size_t MAX_N, const size_t _filter_size,
                           const size_t output_channels) {
  filter = _filter_size;
  channels = output_channels;
  N = MAX_N;
}

CudaConvolve::~CudaConvolve() {
  ReportCUDAErrors(cudaFree(cuda_weights));
  ReportCUDAErrors(cudaFree(cuda_biases));
}

void CudaConvolve::Forward(const int batch, float *input, float *output) {}

void CudaConvolve::LoadingWeight(const std::vector<float> &weights,
                                 const std::vector<float> &biases) {

  const size_t weights_size = sizeof(float) * weights.size();
  const size_t biases_size = sizeof(float) * channels;
  assert(biases_size == sizeof(float) * biases.size() &&
         weights.size() % (filter * filter * channels) == 0);

  ReportCUDAErrors(cudaMalloc(&cuda_weights, weights_size));
  ReportCUDAErrors(cudaMalloc(&cuda_biases, biases_size));

  ReportCUDAErrors(cudaMemcpy(cuda_weights, weights.data(), weights_size,
                              cudaMemcpyHostToDevice));
  ReportCUDAErrors(cudaMemcpy(cuda_biases, biases.data(), biases_size,
                              cudaMemcpyHostToDevice));
}

void cuda_test::test_im2col(const size_t filter_size, const int channels,
                            const std::vector<float> &input,
                            std::vector<float> &output) {
  size_t i_s = input.size() * sizeof(float);
  size_t o_s = output.size() * sizeof(float);
  float *cuda_i;
  float *cuda_o;
  ReportCUDAErrors(cudaMalloc(&cuda_i, i_s));
  ReportCUDAErrors(cudaMalloc(&cuda_o, o_s));

  ReportCUDAErrors(
      cudaMemcpy(cuda_i, input.data(), i_s, cudaMemcpyHostToDevice));

  im2col(filter_size, 1, channels, CONV2D_SIZE, CONV2D_SIZE, cuda_i, cuda_o);

  ReportCUDAErrors(
      cudaMemcpy(output.data(), cuda_o, o_s, cudaMemcpyDeviceToHost));

  ReportCUDAErrors(cudaFree(cuda_i));
  ReportCUDAErrors(cudaFree(cuda_o));
}

void cuda_test::test_batchnorm(const size_t channels, std::vector<float> &data,
                               const std::vector<float> &means,
                               const std::vector<float> &stddevs,
                               const float *const eltwise) {

  size_t m_s = channels * sizeof(float);
  size_t s_s = channels * sizeof(float);
  size_t d_s = data.size() * sizeof(float);
  size_t spatial_size = CONV2D_SIZE * CONV2D_SIZE;
  float *cuda_means;
  float *cuda_stddevs;
  float *cuda_data;
  float *cuda_eltwise;

  ReportCUDAErrors(cudaMalloc(&cuda_means, m_s));
  ReportCUDAErrors(cudaMalloc(&cuda_stddevs, s_s));
  ReportCUDAErrors(cudaMalloc(&cuda_data, d_s));
  ReportCUDAErrors(cudaMalloc(&cuda_eltwise, d_s));

  ReportCUDAErrors(
      cudaMemcpy(cuda_data, data.data(), d_s, cudaMemcpyHostToDevice));
  ReportCUDAErrors(
      cudaMemcpy(cuda_means, means.data(), m_s, cudaMemcpyHostToDevice));
  ReportCUDAErrors(
      cudaMemcpy(cuda_stddevs, stddevs.data(), s_s, cudaMemcpyHostToDevice));

  if (eltwise) {
    ReportCUDAErrors(
        cudaMemcpy(cuda_eltwise, eltwise, d_s, cudaMemcpyHostToDevice));
  } else {
    cuda_eltwise = nullptr;
  }

  batchNorm(cuda_data, cuda_means, cuda_stddevs, 1, channels, spatial_size,
            cuda_eltwise);

  ReportCUDAErrors(
      cudaMemcpy(data.data(), cuda_data, d_s, cudaMemcpyDeviceToHost));

  ReportCUDAErrors(cudaFree(cuda_means));
  ReportCUDAErrors(cudaFree(cuda_stddevs));
  ReportCUDAErrors(cudaFree(cuda_data));
  ReportCUDAErrors(cudaFree(cuda_eltwise));
}

void cuda_test::test_gemm_gpu(bool TA, bool TB, int M, int N, int K,
                              float ALPHA, const float *A, int lda,
                              const float *B, int ldb, float BETA, float *C,
                              int ldc) {

  size_t A_s = M * K * sizeof(float);
  size_t B_s = N * K * sizeof(float);
  size_t C_s = M * N * sizeof(float);
  float *cuda_A;
  float *cuda_B;
  float *cuda_C;

  ReportCUDAErrors(cudaMalloc(&cuda_A, A_s));
  ReportCUDAErrors(cudaMalloc(&cuda_B, B_s));
  ReportCUDAErrors(cudaMalloc(&cuda_C, C_s));

  ReportCUDAErrors(cudaMemcpy(cuda_A, A, A_s, cudaMemcpyHostToDevice));
  ReportCUDAErrors(cudaMemcpy(cuda_B, B, B_s, cudaMemcpyHostToDevice));

  cuda_gemm(TA, TB, M, N, K, ALPHA, cuda_A, lda, cuda_B, ldb, BETA, cuda_C,
            ldc);

  ReportCUDAErrors(cudaMemcpy(C, cuda_C, C_s, cudaMemcpyDeviceToHost));

  ReportCUDAErrors(cudaFree(cuda_A));
  ReportCUDAErrors(cudaFree(cuda_B));
  ReportCUDAErrors(cudaFree(cuda_C));
}

#endif
