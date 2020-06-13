#include "cuda/CUDALayers.h"
#include "cuda/CUDACommon.h"
#include "cuda/CUDAKernels.h"
#include <cassert>
#ifdef USE_CUDA
CudaBatchnorm::CudaBatchnorm(int MAX_batch, int output_channels) {

  channels = output_channels;
  batch = MAX_batch;
  is_loaded = false;
}

CudaBatchnorm::~CudaBatchnorm() {
  if (is_loaded) {
    ReportCUDAErrors(cudaFree(cuda_means));
    ReportCUDAErrors(cudaFree(cuda_stddevs));
  }
}

void CudaBatchnorm::Forward(const int b, float *data,
                            const float *const eltwise) {
  assert(b <= batch);
  cuda_batchnorm(data, cuda_means, cuda_stddevs, b, channels, spatial_size,
                 eltwise);
}


void CudaBatchnorm::cpu_Forward(const int b, std::vector<float> &data, float* eltwise) {

  size_t d_s = data.size() * sizeof(float);
  float *cuda_data;
  float *cuda_eltwise;

  assert(b <= batch);

  ReportCUDAErrors(cudaMalloc(&cuda_data, d_s));
  ReportCUDAErrors(cudaMalloc(&cuda_eltwise, d_s));

  ReportCUDAErrors(
      cudaMemcpy(cuda_data, data.data(), d_s, cudaMemcpyHostToDevice));

  if (eltwise) {
    ReportCUDAErrors(
      cudaMemcpy(cuda_eltwise, eltwise, d_s, cudaMemcpyHostToDevice));
  } else {
    cuda_eltwise = nullptr;
  }

  Forward(b, cuda_data, cuda_eltwise);

  ReportCUDAErrors(
      cudaMemcpy(data.data(), cuda_data, d_s, cudaMemcpyDeviceToHost));

  ReportCUDAErrors(cudaFree(cuda_data));
  ReportCUDAErrors(cudaFree(cuda_eltwise));

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
  is_loaded = true;
}

CudaConvolve::CudaConvolve(const size_t MAX_batch, const size_t filter_size,
                           const size_t input_channels, const size_t output_channels) {

  in_channels = input_channels;
  out_channels = output_channels;
  filter = filter_size;
  filter_dim = filter * filter * in_channels;
  batch = MAX_batch;
  is_loaded = false;
}

CudaConvolve::~CudaConvolve() {
  if (is_loaded) {
    ReportCUDAErrors(cudaFree(cuda_weights));
    ReportCUDAErrors(cudaFree(cuda_biases));
  }
}

void CudaConvolve::Forward(const int b, float *input, float *output) {
  // BUG: 在 filter = 1 時，cuda_im2col 無法得到正確的答案
  // TODO: 支持多 batch 

  assert(b == 1);
  if (filter != 1) {
    cuda_im2col(filter, b, in_channels, height, width, input, cuda_col);
    cuda_gemm(false, false, out_channels, spatial_size, filter_dim, 1.0f,
                 cuda_weights, filter_dim, cuda_col, spatial_size,
                 0.0f, output, spatial_size);
  } else {
    cuda_gemm(false, false, out_channels, spatial_size, filter_dim, 1.0f,
                 cuda_weights, filter_dim, input, spatial_size,
                 0.0f, output, spatial_size);
  }
}

void CudaConvolve::cpu_Forward(const int b, const std::vector<float> &input, std::vector<float> &output) {

  size_t input_s = input.size() * sizeof(float);
  size_t output_s = output.size() * sizeof(float);
  float *cuda_input;
  float *cuda_output;

  ReportCUDAErrors(cudaMalloc(&cuda_input, input_s));
  ReportCUDAErrors(cudaMalloc(&cuda_output, output_s));

  ReportCUDAErrors(
      cudaMemcpy(cuda_input, input.data(), input_s, cudaMemcpyHostToDevice));

  Forward(b, cuda_input, cuda_output);

  ReportCUDAErrors(
      cudaMemcpy(output.data(), cuda_output, output_s, cudaMemcpyDeviceToHost));

  ReportCUDAErrors(cudaFree(cuda_input));
  ReportCUDAErrors(cudaFree(cuda_output));
}


void CudaConvolve::LoadingWeight(const std::vector<float> &weights,
                                 const std::vector<float> &biases) {

  const size_t weights_size = sizeof(float) * weights.size();
  const size_t biases_size = sizeof(float) * out_channels;
  assert(biases_size == sizeof(float) * biases.size() &&
         weights.size() % (filter * filter * out_channels * in_channels) == 0);

  w_s = weights_size;

  ReportCUDAErrors(cudaMalloc(&cuda_weights, weights_size));
  ReportCUDAErrors(cudaMalloc(&cuda_biases, biases_size));
  ReportCUDAErrors(cudaMalloc(&cuda_col, weights_size * filter * filter));

  ReportCUDAErrors(cudaMemcpy(cuda_weights, weights.data(), weights_size,
                              cudaMemcpyHostToDevice));
  ReportCUDAErrors(cudaMemcpy(cuda_biases, biases.data(), biases_size,
                              cudaMemcpyHostToDevice));
  is_loaded = true;
}
#endif
