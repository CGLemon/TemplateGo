#ifndef CUDABLAS_H_INCLUDE
#define CUDABLAS_H_INCLUDE
#ifdef USE_CUDA
#include <cassert>
#include "cuda/CUDACommon.h"

template <typename T>
void cuda_add_vectors(T *c, T *a, T *b, int size, int asize, int bsize, bool relu);

template <typename T>
void cuda_batchnorm(T *data, const float *means, const float *stddevs,
                    int batch, int channels, int spatial_size,
                    const T *eltwise, bool relu);

template <typename T>
void cuda_im2col(int filter_size, int channels, int H, int W,
                 T *data_im, T *data_col);

template<typename T>
void cuda_global_avg_pool(T *input, T *output, int batch, int channels, int spatial_size);

template<typename T>
void cuda_se_scale(const T *input, const T* se_bias, T* data,
                   int batch, int channels, int spatial_size);

template<typename T>
void cuda_input_pool(const T *bias, T *data,
                     int batch, int channels, int spatial_size);


void cuda_gemm(bool TA, bool TB, int M, int N, int K, float ALPHA,
               const float *A_gpu, int lda, const float *B_gpu, int ldb,
               float BETA, float *C_gpu, int ldc, cublasHandle_t * handle);

template<typename T>
void cuda_swap(T *a, T *b, int size);

template<typename T>
void cuda_copy(T *a, T *b, int size);

#endif
#endif
