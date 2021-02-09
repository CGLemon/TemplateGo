#ifndef CUDALAYER_H_INCLUDE
#define CUDALAYER_H_INCLUDE
#ifdef USE_CUDA
#include "cuda/CUDACommon.h"
#include "Winograd_helper.h"
#include "config.h"

#include <vector>
#include <array>

class CudaBatchnorm {
public:
    CudaBatchnorm() = default;
    CudaBatchnorm(const size_t conv_size, const size_t batch,
                  const size_t channels, bool ReLU = true);
    ~CudaBatchnorm();

    void Forward(const size_t batch, float *data,
                 const float *const eltwise = nullptr);

    void LoadingWeight(const std::vector<float> &means,
                       const std::vector<float> &stddevs);

    void set_convsize(const size_t conv_size);

private:
    int spatial_size;
    int m_channels;
    int m_batch;

    bool m_ReLU;
    bool is_loaded{false};
    float *cuda_means;
    float *cuda_stddevs;
};


class CudaConvolve {
public:
    CudaConvolve() = default;
    CudaConvolve(const size_t conv_size, const size_t batch,
                 const size_t filter, const size_t in_channels, const size_t out_channels);
    ~CudaConvolve();

    void Forward(const int batch, float *input, float *output,
                 void *scratch, size_t scratch_size, CudaHandel *handel);

    void LoadingWeight(const std::vector<float> &weights, size_t &scratch_size);

    void set_convsize(const size_t conv_size);

private:
    int width;
    int height;
    int spatial_size;
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
    // float *cuda_col;
};

class CudaFullyConnect {
public:
    CudaFullyConnect() = default;
    CudaFullyConnect(const size_t batch, const size_t inputs, 
                     const size_t outputs, bool ReLU);
    ~CudaFullyConnect();

    void Forward(const int batch,
                 float *input,
                 float *output,
                 CudaHandel *handel);


    void LoadingWeight(const std::vector<float> &weights,
                       const std::vector<float> &biases);

    void set_size(const size_t in_size, const size_t out_size);

private:
    bool m_ReLU;
    int m_batch;
    int m_inputs;
    int m_outputs;

    bool is_loaded{false};
    float *cuda_weights;
    float *cuda_biases;
};

class CudaGlobalAvgPool {
public:
    CudaGlobalAvgPool() = default; 
    CudaGlobalAvgPool(const size_t conv_size,
                      const size_t batch,
                      const size_t channels);

    void Forward(const int batch, float *input, float *output);

    void set_convsize(const size_t conv_size);

private:
    int width;
    int height;
    int spatial_size;
    int m_batch;
    int m_channels;
};

class CudaSEUnit {
public:
    CudaSEUnit() = default;
    CudaSEUnit(const size_t conv_size, const size_t batch,
               const size_t channels, const size_t se_size);
    ~CudaSEUnit();

    void LoadingWeight(const std::vector<float> &weights_w1,
                       const std::vector<float> &weights_b1,
                       const std::vector<float> &weights_w2,
                       const std::vector<float> &weights_b2);

    void Forward(const int batch, float *input, float *output, CudaHandel *handel);

    void set_convsize(const size_t conv_size);
 
private:
    int width;
    int height;
    int spatial_size;

    int m_se_size;
    int m_batch;
    int m_channels;

    bool is_loaded{false};
    std::array<float *, 3> cuda_op;

    float *cuda_weights_w1;
    float *cuda_weights_b1;
    float *cuda_weights_w2;
    float *cuda_weights_b2;
};

class CudaInputPool {
public:
    CudaInputPool() = default;
    CudaInputPool(const size_t conv_size, const size_t batch,
                  const size_t input_size, const size_t channels);
    ~CudaInputPool();

    void LoadingWeight(const std::vector<float> &weights_w,
                      const std::vector<float> &weights_b); 

    void Forward(const int batch, float *input, float *output, CudaHandel *handel);

    void set_convsize(const size_t conv_size);
 
private:
    int width;
    int height;
    int spatial_size;

    int m_batch;
    int m_input_size;
    int m_channels;

    bool is_loaded{false};

    float *cuda_op;
    float *cuda_weights_w;
    float *cuda_weights_b;

};

#endif
#endif
