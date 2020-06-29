#include "cuda/CUDAKernels.h"
#ifdef USE_CUDA
template <typename T>
__global__ void cuda_addVectors_kernel(T *a, T *b, T *c, int asize, int bsize,
                                       int csize, bool relu) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < size) {
    float aVal = 0;
    float bVal = 0;
    if (a) aVal = (float)(a[i % asize]);
    if (b) bVal = (float)(b[i % bsize]);

    float cVal = aVal + bVal;

    if (relu && (cVal < 0)) cVal = 0;
    c[i] = (T)cVal;
  }
}

template <typename T>
void cuda_addVectors(T *a, T *b, T *c, int asize, int bsize,
                     int csize, bool relu) {
  const int kBlockSize = KBLOCKSIZE;
  const int blocks = DivUp(size, kBlockSize);

  cuda_addVectors_kernel<<<blocks, kBlockSize>>>(a, b, c, asize, bsize, size, relu);
  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void batchNorm_kernel(T *data, const float *means,
                                 const float *stddevs, int batch, int C,
                                 int spatial_size, const T *eltwise) {

  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int size = batch * C * spatial_size;
  if (index < size) {
    int wIndex = (index / (spatial_size)) % C;

    float el = data[index];
    float mean = means[wIndex];
    float scale_stddev = stddevs[wIndex];

    el -= mean;
    el *= scale_stddev;

    if (eltwise)
      el += (float)eltwise[index];

    if ((el < 0))
      el = 0;

    data[index] = (T)el;
  }
}

template <typename T>
void cuda_batchnorm(T *data, const float *means, const float *stddevs,
                    int batch, int channels, int spatial_size,
                    const T *eltwise) {

  const int total_elements = batch * channels * spatial_size;
  const int kBlockSize = KBLOCKSIZE;
  const int blocks = DivUp(total_elements, kBlockSize);

  batchNorm_kernel<<<blocks, kBlockSize>>>(data, means, stddevs, batch,
                                           channels, spatial_size, eltwise);

  ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void im2col_kernel(int filter_size, int pad, int batch, int C, int H,
                              int W, int output_h, int output_w, T *data_im,
                              T *data_col) {
  int total_elements = batch * C * H * W;
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < total_elements) {
    int CWH_size = C * W * H;
    int Windex = index % CWH_size;
    int n = index / CWH_size;

    assert(n == 0);

    int w_out = Windex % output_w;
    int h_index = Windex / output_w;
    int h_out = h_index % output_h;

    int channel_in = h_index / output_h;
    int channel_out = channel_in * filter_size * filter_size;

    int h_in = h_out - pad;
    int w_in = w_out - pad;

    float *data_col_ptr = data_col;
    data_col_ptr +=
        CWH_size * n + (channel_out * output_h + h_out) * output_w + w_out;
    const float *data_im_ptr = data_im;
    data_im_ptr += CWH_size * n + (channel_in * H + h_in) * W + w_in;

    for (int kernel_row = 0; kernel_row < filter_size; kernel_row++) {
      for (int kernel_col = 0; kernel_col < filter_size; kernel_col++) {
        int h = h_in + kernel_row;
        int w = w_in + kernel_col;
        *data_col_ptr = (h >= 0 && w >= 0 && h < H && w < W)
                            ? data_im_ptr[kernel_row * W + kernel_col]
                            : 0;
        data_col_ptr += output_w * output_h;
      }
    }
  }
}

template <typename T>
void cuda_im2col(int filter_size, int batch, int channels, int H, int W,
                 T *input, T *output) {

  const int total_elements = batch * channels * H * W;
  const int kBlockSize = KBLOCKSIZE;
  const int blocks = DivUp(total_elements, kBlockSize);

  const int pad = (filter_size / 2);
  const int output_h = H + 2 * pad - filter_size + 1;
  const int output_w = W + 2 * pad - filter_size + 1;

  im2col_kernel<<<blocks, kBlockSize>>>(filter_size, pad, batch, channels, H, W,
                                        output_h, output_w, input, output);
  ReportCUDAErrors(cudaGetLastError());
}

void cuda_gemm(bool TA, bool TB, int M, int N, int K, float ALPHA,
               const float *A_gpu, int lda, const float *B_gpu, int ldb,
               float BETA, float *C_gpu, int ldc, cublasHandle_t handle) {
  //cublasHandle_t cublas = blas_handle();
  ReportCUBLASErrors(cublasSgemm(handle, (TB ? CUBLAS_OP_T : CUBLAS_OP_N),
                                 (TA ? CUBLAS_OP_T : CUBLAS_OP_N), N, M, K,
                                 &ALPHA, B_gpu, ldb, A_gpu, lda, &BETA, C_gpu,
                                 ldc));
}

template <typename T>
__global__ void swap_kernel(T *a, T *b, int size) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < size) {
    T temp_a = a[index];
    T temp_b = b[index];
    a[index] = temp_b;
    b[index] = temp_a;
  }
} 

template <typename T>
void cuda_swap(T *a, T *b, int size) {
  const int kBlockSize = KBLOCKSIZE;
  const int blocks = DivUp(size, kBlockSize);
  swap_kernel<<<blocks, kBlockSize>>>(a, b, size);
}


template <typename T>
__global__ void copy_kernel(T *a, T *b, int size) {
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  if (index < size) {
    a[index] = b[index];
  }
} 


template<typename T>
void cuda_copy(T *a, T *b, int size) {
  const int kBlockSize = KBLOCKSIZE;
  const int blocks = DivUp(size, kBlockSize);
  copy_kernel<<<blocks, kBlockSize>>>(a, b, size);
}



template void cuda_batchnorm<float>(float *data, const float *means,
                                    const float *stddevs, int N, int channels,
                                    int spatial_size, const float *eltwise);

template void cuda_im2col<float>(int filter_size, int N, int C, int H, int W,
                                 float *data_im, float *data_col);

template void cuda_swap<float>(float *a, float *b, int size);

template void cuda_copy<float>(float *a, float *b, int size);

template void cuda_addVectors<float>(float *c, float *a, float *b, int size, int asize, int bsize, bool relu);
#endif
