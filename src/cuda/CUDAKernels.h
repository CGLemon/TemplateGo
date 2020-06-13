#ifndef CUDABLAS_H_INCLUDE
#define CUDABLAS_H_INCLUDE
#ifdef USE_CUDA
#include <cassert>

// Perform batch normilization.
template <typename T>
void cuda_batchnorm(T *data, const float *means, const float *stddevs,
                    int batch, int channels, int spatial_size,
                    const T *eltwise);

template <typename T>
void cuda_im2col(int filter_size, int batch, int channels, int H, int W,
                 T *data_im, T *data_col);

void cuda_gemm(bool TA, bool TB, int M, int N, int K, float ALPHA,
               const float *A_gpu, int lda, const float *B_gpu, int ldb,
               float BETA, float *C_gpu, int ldc);


template<typename T>
void cuda_swap(T *a, T *b, int size);

#endif
#endif
