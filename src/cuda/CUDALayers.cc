#include "cuda/CUDALayers.h"
#include "cuda/CUDAKernels.h"
#include <cassert>
#include <algorithm> 
#ifdef USE_CUDA

CudaBatchnorm::CudaBatchnorm(const size_t conv_size,
                             const size_t MAX_batch, const size_t output_channels, bool ReLU) {

    m_channels = output_channels;
    m_batch = MAX_batch;
    spatial_size = conv_size * conv_size;
    is_loaded = false;
    m_ReLU = ReLU;
}

CudaBatchnorm::~CudaBatchnorm() {
    if (is_loaded) {
        ReportCUDAErrors(cudaFree(cuda_means));
        ReportCUDAErrors(cudaFree(cuda_stddevs));
    }
}

void CudaBatchnorm::set_convsize(const size_t conv_size) {
    spatial_size = conv_size * conv_size;
}

void CudaBatchnorm::Forward(const size_t batch, float *data,
                            const float *const eltwise) {
    if (!is_loaded) {
        return;
    }

    assert(batch <= m_batch);
    cuda_batchnorm(data, cuda_means, cuda_stddevs,
                   batch, m_channels, spatial_size, eltwise, m_ReLU);
}


void CudaBatchnorm::LoadingWeight(const std::vector<float> &means,
                                  const std::vector<float> &stddevs) {
    if (is_loaded) {
        return;
    }

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


CudaConvolve::CudaConvolve(const size_t conv_size, const size_t MAX_batch,
                           const size_t filter_size, const size_t input_channels, const size_t output_channels) {

    m_in_channels = input_channels;
    m_out_channels = output_channels;
    m_filter = filter_size;
    m_filter_dim = m_filter * m_filter * m_in_channels;
    m_batch = MAX_batch;
    width = conv_size;
    height = conv_size;
    spatial_size = width * height;

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

void CudaConvolve::set_convsize(const size_t conv_size) {
    width = conv_size;
    height = conv_size;
    spatial_size = width * height;
}

void CudaConvolve::Forward(const int batch, float *input, float *output,
                           void *scratch, size_t scratch_size, CudaHandel *handel) {
    if (!is_loaded) {
        return;
    }
    assert(batch <= m_batch);
#ifdef USE_CUDNN
    if (!scratch) {
        return;
    }

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
                      conv_desc, conv_algo, scratch, scratch_size, &beta, out_tensor_desc,
                      output));
#else
    auto op_scratch = reinterpret_cast<float*>(scratch);
    const size_t input_shift = m_in_channels * spatial_size;
    const size_t output_shift = m_out_channels * spatial_size;
    for (auto b = size_t{0}; b < batch; ++b) {
        float *input_ptr = input + b * input_shift;
        float *output_ptr = output + b * output_shift;
        if (m_filter != 1) {
            cuda_im2col(m_filter, m_in_channels, height, width, input_ptr, op_scratch);
            cuda_gemm(false, false, m_out_channels, spatial_size, m_filter_dim, 1.0f,
                      cuda_weights, m_filter_dim, op_scratch, spatial_size,
                      0.0f, output_ptr, spatial_size, &handel->cublas_handel);
        } else {
            cuda_gemm(false, false, m_out_channels, spatial_size, m_filter_dim, 1.0f,
                      cuda_weights, m_filter_dim, input_ptr, spatial_size,
                      0.0f, output_ptr, spatial_size, &handel->cublas_handel);
        }
    }
#endif
}

void CudaConvolve::LoadingWeight(const std::vector<float> &weights, size_t &scratch_size) {

    if (is_loaded) {
        return;
    }
    const size_t weights_size = sizeof(float) * weights.size();
    assert(weights.size() == (m_filter_dim * m_out_channels));

    ReportCUDAErrors(cudaMalloc(&cuda_weights, weights_size));
    ReportCUDAErrors(cudaMemcpy(cuda_weights, weights.data(), weights_size,
                                cudaMemcpyHostToDevice));
    is_loaded = true;
    size_t apply_scratch_size = 0;
#ifdef USE_CUDNN
    if (cudnn_applied) { 
        return;
    }

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
                                                 1, m_in_channels, height, width));

    ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc,
                                                 CUDNN_TENSOR_NCHW,
                                                 CUDNN_DATA_FLOAT,
                                                 1, m_out_channels, height, width));

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

    cudnn_applied = true;
    const size_t max_scratch_size = std::max(apply_scratch_size, scratch_size);
    scratch_size = max_scratch_size;
#else
    apply_scratch_size = weights_size * m_filter * m_filter;
    const size_t max_scratch_size = std::max(apply_scratch_size, scratch_size);
    scratch_size = max_scratch_size;
#endif
    
}

CudaFullyConnect::CudaFullyConnect(const size_t MAX_batch, const size_t inputs, 
                                   const size_t outputs, bool ReLU) {
    m_batch = MAX_batch;
    m_inputs = inputs;
    m_outputs = outputs;
    is_loaded = false;
    m_ReLU = ReLU;
}

void CudaFullyConnect::set_size(const size_t in_size, const size_t out_size) {
    m_inputs = in_size;
    m_outputs = out_size;
}

CudaFullyConnect::~CudaFullyConnect() {
    if (is_loaded) {
        ReportCUDAErrors(cudaFree(cuda_weights));
        ReportCUDAErrors(cudaFree(cuda_biases));
    }
}

void CudaFullyConnect::LoadingWeight(const std::vector<float> &weights,
                                     const std::vector<float> &biases) {

    if (is_loaded) { 
        return;
    }
    const size_t weights_size = sizeof(float) * weights.size();
    const size_t biases_size = sizeof(float) * biases.size();

    assert(weights.size() == m_inputs * m_outputs);
    assert(biases.size() == m_outputs);

    ReportCUDAErrors(cudaMalloc(&cuda_weights, weights_size));
    ReportCUDAErrors(cudaMalloc(&cuda_biases, biases_size));
  
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights, weights.data(), weights_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_biases, biases.data(), biases_size, cudaMemcpyHostToDevice));
    is_loaded = true;
}

void CudaFullyConnect::Forward(const int batch, float *input, float *output, CudaHandel *handel) {

    if (!is_loaded) {
        return;
    }
    assert(batch <= m_batch);
    cuda_gemm(false, true,
             batch,
             m_outputs,
             m_inputs,
             1.0f,
             input,
             m_inputs, 
             cuda_weights,
             m_inputs,
             0.0f,
             output,
             m_outputs,
             &handel->cublas_handel);

    cuda_add_vectors(output, cuda_biases, output, m_outputs * batch,
                     m_outputs,  m_outputs * batch, m_ReLU);
}

CudaGlobalAvgPool::CudaGlobalAvgPool(const size_t conv_size,
                                     const size_t MAX_batch,
                                     const size_t channels) {
    width = conv_size;
    height = conv_size;
    spatial_size = width * height;
    m_batch = MAX_batch;
    m_channels = channels;
}

void CudaGlobalAvgPool::Forward(const int batch, float *input, float *output) {
    cuda_global_avg_pool(input, output, batch,
                         m_channels, spatial_size);
}

void CudaGlobalAvgPool::set_convsize(const size_t conv_size) {
    width = conv_size;
    height = conv_size;
    spatial_size = width * height;
}


CudaSEUnit::CudaSEUnit(const size_t conv_size, const size_t MAX_batch,
                       const size_t channels, const size_t se_size) {
    width = conv_size;
    height = conv_size;
    spatial_size = width * height;

    m_se_size = se_size;
    m_batch = MAX_batch;
    m_channels = channels;
    is_loaded = false;
}

void CudaSEUnit::LoadingWeight(const std::vector<float> &weights_w1,
                               const std::vector<float> &weights_b1,
                               const std::vector<float> &weights_w2,
                               const std::vector<float> &weights_b2) {

    if (is_loaded) { 
        return;
    }
    const size_t type_size = sizeof(float);
    const size_t weights_w1_size = type_size * weights_w1.size();
    const size_t weights_b1_size = type_size * weights_b1.size();
    const size_t weights_w2_size = type_size * weights_w2.size();
    const size_t weights_b2_size = type_size * weights_b2.size();

    assert(weights_w1.size() == m_channels * m_se_size);
    assert(weights_b1.size() == m_se_size);
    assert(weights_w2.size() == 2 * m_se_size * m_channels);
    assert(weights_b2.size() == 2 * m_channels);

    ReportCUDAErrors(cudaMalloc(&cuda_weights_w1, weights_w1_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_b1, weights_b1_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_w2, weights_w2_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_b2, weights_b2_size));

    const size_t fc1_scratch_size = type_size * m_batch * m_se_size;
    const size_t fc2_scratch_size = type_size * 2 * m_batch * m_channels;
    const size_t pool_scratch_size = type_size * m_batch * m_channels;

    ReportCUDAErrors(cudaMalloc(&cuda_op[0], pool_scratch_size));
    ReportCUDAErrors(cudaMalloc(&cuda_op[1], fc1_scratch_size));
    ReportCUDAErrors(cudaMalloc(&cuda_op[2], fc2_scratch_size));

    is_loaded = true;

    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_w1, weights_w1.data(), weights_w1_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_b1, weights_b1.data(), weights_b1_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_w2, weights_w2.data(), weights_w2_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_b2, weights_b2.data(), weights_b2_size, cudaMemcpyHostToDevice));
}

void CudaSEUnit::Forward(const int batch, float *input, float *ouput, CudaHandel *handel) {

    cuda_global_avg_pool(input, cuda_op[0], batch, m_channels, spatial_size);


    const size_t fc1_input_size = m_channels;
    const size_t fc1_output_size = m_se_size;
    const bool fc1_relu = true;
    cuda_gemm(false, true,
              batch,
              fc1_output_size,
              fc1_input_size, 
              1.0f,
              cuda_op[0],
              fc1_input_size, 
              cuda_weights_w1,
              fc1_input_size,
              0.0f,
              cuda_op[1],
              fc1_output_size,
              &handel->cublas_handel);

    cuda_add_vectors(cuda_op[1], cuda_weights_b1, cuda_op[1], fc1_output_size * batch,
                     fc1_output_size,  fc1_output_size * batch, fc1_relu);

    const size_t fc2_input_size = m_se_size;
    const size_t fc2_output_size = 2 * m_channels;
    const bool fc2_relu = false;
    cuda_gemm(false, true,
              batch,
              fc2_output_size,
              fc2_input_size, 
              1.0f,
              cuda_op[1],
              fc2_input_size, 
              cuda_weights_w2,
              fc2_input_size,
              0.0f,
              cuda_op[2],
              fc2_output_size,
              &handel->cublas_handel);

    cuda_add_vectors(cuda_op[2], cuda_weights_b2, cuda_op[2], fc2_output_size * batch,
                     fc2_output_size,  fc2_output_size * batch, fc2_relu);

    cuda_se_scale(input, cuda_op[2], ouput, batch, m_channels, spatial_size);
}

CudaSEUnit::~CudaSEUnit() {
    if (is_loaded) {
        ReportCUDAErrors(cudaFree(cuda_weights_w1));
        ReportCUDAErrors(cudaFree(cuda_weights_b1));
        ReportCUDAErrors(cudaFree(cuda_weights_w2));
        ReportCUDAErrors(cudaFree(cuda_weights_b2));

        ReportCUDAErrors(cudaFree(cuda_op[0]));
        ReportCUDAErrors(cudaFree(cuda_op[1]));
        ReportCUDAErrors(cudaFree(cuda_op[2]));
    }
}

void CudaSEUnit::set_convsize(size_t conv_size) {
    width = conv_size;
    height = conv_size;
    spatial_size = width * height;
}

CudaInputPool::CudaInputPool(const size_t conv_size, const size_t MAX_batch,
                             const size_t input_size, const size_t channels) {
    width = conv_size;
    height = conv_size;
    spatial_size = width * height;

    m_input_size = input_size;
    m_batch = MAX_batch;
    m_channels = channels;
    is_loaded = false;
}

void CudaInputPool::LoadingWeight(const std::vector<float> &weights_w,
                                  const std::vector<float> &weights_b) {

    if (is_loaded) { 
        return;
    }
    const size_t type_size = sizeof(float);
    const size_t weights_w_size = type_size * weights_w.size();
    const size_t weights_b_size = type_size * weights_b.size();
    assert(weights_w.size() == m_input_size * m_channels);
    assert(weights_b.size() == m_channels);

    ReportCUDAErrors(cudaMalloc(&cuda_weights_w, weights_w_size));
    ReportCUDAErrors(cudaMalloc(&cuda_weights_b, weights_b_size));

    const size_t fc_scratch_size = type_size * m_batch * m_channels;

    ReportCUDAErrors(cudaMalloc(&cuda_op, fc_scratch_size));

    is_loaded = true;

    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_w, weights_w.data(), weights_w_size, cudaMemcpyHostToDevice));
    ReportCUDAErrors(cudaMemcpy(
        cuda_weights_b, weights_b.data(), weights_b_size, cudaMemcpyHostToDevice));
}

void CudaInputPool::Forward(const int batch, float *input, float *output, CudaHandel *handel) {

    const size_t fc_input_size = m_input_size;
    const size_t fc_output_size = m_channels;
    const bool fc_relu = false;
    cuda_gemm(false, true,
              batch,
              fc_output_size,
              fc_input_size, 
              1.0f,
              input,
              fc_input_size, 
              cuda_weights_w,
              fc_input_size,
              0.0f,
              cuda_op,
              fc_output_size,
              &handel->cublas_handel);

    cuda_add_vectors(cuda_op, cuda_weights_b, cuda_op, fc_output_size * batch,
                     fc_output_size,  fc_output_size * batch, fc_relu);


    cuda_input_pool(cuda_op, output,
                    batch, m_channels, spatial_size);
}

void CudaInputPool::set_convsize(const size_t conv_size) {
    width = conv_size;
    height = conv_size;
    spatial_size = width * height;
}

CudaInputPool::~CudaInputPool() {
    if (is_loaded) {
        ReportCUDAErrors(cudaFree(cuda_weights_w));
        ReportCUDAErrors(cudaFree(cuda_weights_b));
        ReportCUDAErrors(cudaFree(cuda_op));
    }
}

#endif
