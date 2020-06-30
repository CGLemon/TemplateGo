#include "cuda/CUDALayers.h"
#include "cuda/CUDAKernels.h"
#include <cassert>
#ifdef USE_CUDA
CudaBatchnorm::CudaBatchnorm(int MAX_batch, int output_channels) {

  m_channels = output_channels;
  m_batch = MAX_batch;
  is_loaded = false;
}

CudaBatchnorm::~CudaBatchnorm() {
  if (is_loaded) {
    ReportCUDAErrors(cudaFree(cuda_means));
    ReportCUDAErrors(cudaFree(cuda_stddevs));
  }
}

void CudaBatchnorm::Forward(const size_t batch, float *data,
                            const float *const eltwise) {

  if (!is_loaded) return;
  assert(batch <= m_batch);
  cuda_batchnorm(data, cuda_means, cuda_stddevs, batch, m_channels, spatial_size,
                 eltwise);
}


void CudaBatchnorm::cpu_Forward(const int batch, std::vector<float> &data,
                                float* eltwise) {

  assert(batch <= m_batch);

  size_t d_s = batch * data.size() * sizeof(float);
  float *cuda_data;
  float *cuda_eltwise;

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

  Forward(batch, cuda_data, cuda_eltwise);

  ReportCUDAErrors(
      cudaMemcpy(data.data(), cuda_data, d_s, cudaMemcpyDeviceToHost));

  ReportCUDAErrors(cudaFree(cuda_data));
  ReportCUDAErrors(cudaFree(cuda_eltwise));

}


void CudaBatchnorm::LoadingWeight(const std::vector<float> &means,
                                  const std::vector<float> &stddevs) {

  if (is_loaded) return;
  const size_t weights_size = sizeof(float) * m_channels;
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

  m_in_channels = input_channels;
  m_out_channels = output_channels;
  m_filter = filter_size;
  m_filter_dim = m_filter * m_filter * m_in_channels;
  m_batch = MAX_batch;

#ifdef USE_CUDNN
  cudnn_applied = false;
#endif
  is_loaded = false;
}

CudaConvolve::~CudaConvolve() {
  if (is_loaded) {
    ReportCUDAErrors(cudaFree(cuda_weights));
  }

#ifdef USE_CUDNN
  if (cudnn_applied) {
    cudnnDestroyFilterDescriptor(filter_desc);
    cudnnDestroyConvolutionDescriptor(conv_desc);
    cudnnDestroyTensorDescriptor(in_tensor_desc);
    cudnnDestroyTensorDescriptor(out_tensor_desc);
  }
#endif
}

void CudaConvolve::Forward(const int batch, float *input, float *output,
                           void * cuda_scratch, size_t scratch_size, CudaHandel * handel) {
  if (!is_loaded) return;
  assert(batch <= m_batch);
#ifdef USE_CUDNN
  if (!cuda_scratch) return;
  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               batch, m_in_channels, height, width));
  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               batch, m_out_channels, height, width));
  float alpha = 1.0f, beta = 0.0f;
  ReportCUDNNErrors(cudnnConvolutionForward(
                    handel->cudnn_handel, &alpha, in_tensor_desc, input, filter_desc, cuda_weights,
                    conv_desc, conv_algo, cuda_scratch, scratch_size, &beta, out_tensor_desc,
                    output));
#else
  const size_t input_shift = m_in_channels * spatial_size;
  const size_t output_shift = m_out_channels * spatial_size;
  for (auto b = size_t{0}; b < batch; ++b) {
    float * input_ptr = input + b * input_shift;
    float * output_ptr = output + b * output_shift;
    if (m_filter != 1) {
      cuda_im2col(m_filter, m_in_channels, height, width, input_ptr, cuda_col);
      cuda_gemm(false, false, m_out_channels, spatial_size, m_filter_dim, 1.0f,
                cuda_weights, m_filter_dim, cuda_col, spatial_size,
                0.0f, output_ptr, spatial_size, &handel->cublas_handel);
    } else {
      cuda_gemm(false, false, m_out_channels, spatial_size, m_filter_dim, 1.0f,
                cuda_weights, m_filter_dim, input_ptr, spatial_size,
                0.0f, output_ptr, spatial_size, &handel->cublas_handel);
    }
  }
#endif

}

void CudaConvolve::cpu_Forward(const int batch, const std::vector<float> &input,
                               std::vector<float> &output) {
  
  size_t input_s = batch * input.size() * sizeof(float);
  size_t output_s = batch * output.size() * sizeof(float);
  float *cuda_input;
  float *cuda_output;

  ReportCUDAErrors(cudaMalloc(&cuda_input, input_s));
  ReportCUDAErrors(cudaMalloc(&cuda_output, output_s));

  ReportCUDAErrors(
      cudaMemcpy(cuda_input, input.data(), input_s, cudaMemcpyHostToDevice));
  
  CudaHandel handel;
  handel.apply();
  void * cuda_scratch = nullptr;
  size_t scratch_size = 0;

  Forward(batch, cuda_input, cuda_output, cuda_scratch, scratch_size, &handel);

  ReportCUDAErrors(
      cudaMemcpy(output.data(), cuda_output, output_s, cudaMemcpyDeviceToHost));

  ReportCUDAErrors(cudaFree(cuda_input));
  ReportCUDAErrors(cudaFree(cuda_output));
}


void CudaConvolve::LoadingWeight(const std::vector<float> &weights, size_t & scratch_size) {

  if (is_loaded) return;
  const size_t weights_size = sizeof(float) * weights.size();
  assert(weights.size() 
             == (m_filter_dim * m_out_channels));

  ReportCUDAErrors(cudaMalloc(&cuda_weights, weights_size));
  ReportCUDAErrors(cudaMalloc(&cuda_col, weights_size * m_filter * m_filter));

  ReportCUDAErrors(cudaMemcpy(cuda_weights, weights.data(), weights_size,
                              cudaMemcpyHostToDevice));
  is_loaded = true;
#ifdef USE_CUDNN
  if (cudnn_applied) return;

  size_t apply_scratch_size = 0;
  

  auto cudnn = cudnn_handle();
  cudnnCreateFilterDescriptor(&filter_desc);
  cudnnCreateConvolutionDescriptor(&conv_desc);
  cudnnCreateTensorDescriptor(&out_tensor_desc);
  cudnnCreateTensorDescriptor(&in_tensor_desc);

  ReportCUDNNErrors(cudnnSetFilter4dDescriptor(filter_desc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW,
                                               m_out_channels, m_in_channels, m_filter, m_filter));
  
  const size_t padding = m_filter / 2;
  ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
                    conv_desc, padding, padding, 1, 1, 1, 1,
                    CUDNN_CROSS_CORRELATION, CUDNN_DATA_FLOAT));

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               m_batch, m_in_channels, height, width));
  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc,
                                               CUDNN_TENSOR_NCHW,
                                               CUDNN_DATA_FLOAT,
                                               m_batch, m_out_channels, height, width));

  ReportCUDNNErrors(cudnnGetConvolutionForwardAlgorithm(cudnn,
                                                        in_tensor_desc,
                                                        filter_desc, 
                                                        conv_desc,
                                                        out_tensor_desc, 
                                                        CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
                                                        0,
                                                        &conv_algo));

  ReportCUDNNErrors(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                            in_tensor_desc,
                                                            filter_desc, 
                                                            conv_desc,
                                                            out_tensor_desc, 
                                                            conv_algo, 
                                                            &apply_scratch_size));
  const size_t max_scratch_size = std::max(apply_scratch_size, scratch_size);
  scratch_size = max_scratch_size;
  cudnn_applied = true;
#endif
}

CudaFullyConnect::CudaFullyConnect(const size_t MAX_batch, const size_t inputs, 
                                   const size_t outputs, bool is_relu) {
  m_batch = MAX_batch;
  m_inputs = inputs;
  m_outputs = outputs;
  is_loaded = false;
  m_is_relu = is_relu;
}

void CudaFullyConnect::LoadingWeight(const std::vector<float> &weights,
                                     const std::vector<float> &biases) {

  if (is_loaded) return;
  const size_t weights_size = sizeof(float) * weights.size();
  const size_t biases_size = sizeof(float) * biases.size();

  assert(weights.size() == m_inputs * m_outputs);
  assert(biases.size() == m_outputs);

  ReportCUDAErrors(cudaMalloc(&cuda_weights, weights_size));
  ReportCUDAErrors(cudaMalloc(&cuda_biases, biases_size));
  
  ReportCUDAErrors(cudaMemcpy(cuda_weights, weights.data(), weights_size,
                              cudaMemcpyHostToDevice));
  ReportCUDAErrors(cudaMemcpy(cuda_biases, biases.data(), biases_size,
                              cudaMemcpyHostToDevice));
  is_loaded = true;
}


void CudaFullyConnect::Forward(const int batch, float *input, float *output, CudaHandel * handel) {

  if (!is_loaded) return;
  assert(batch <= m_batch);
  cuda_gemm(false, true, batch, m_outputs, m_inputs, 1.0f, input, m_inputs, 
            cuda_weights, m_inputs, 0.0f, output, m_outputs, &handel->cublas_handel);

  cuda_addVectors(output, cuda_biases, output, m_outputs * batch,
                  m_outputs,  m_outputs * batch, m_is_relu);
}
  

void CudaFullyConnect::cpu_Forward(const int batch, const std::vector<float> &input, 
                                   std::vector<float> &output) {

  const size_t input_s = batch * input.size() * sizeof(float);
  const size_t output_s = batch * output.size() * sizeof(float);
  float *cuda_input;
  float *cuda_output;
  ReportCUDAErrors(cudaMalloc(&cuda_input, input_s));
  ReportCUDAErrors(cudaMalloc(&cuda_output, output_s));

  ReportCUDAErrors(
      cudaMemcpy(cuda_input, input.data(), input_s, cudaMemcpyHostToDevice));

  CudaHandel handel;
  handel.apply();
  Forward(batch, cuda_input, cuda_output, &handel);

  ReportCUDAErrors(
      cudaMemcpy(output.data(), cuda_output, output_s, cudaMemcpyDeviceToHost));

  ReportCUDAErrors(cudaFree(cuda_input));
  ReportCUDAErrors(cudaFree(cuda_output));
}


CudaFullyConnect::~CudaFullyConnect() {
  if (is_loaded) {
    ReportCUDAErrors(cudaFree(cuda_weights));
    ReportCUDAErrors(cudaFree(cuda_biases));
  }
}

#endif
