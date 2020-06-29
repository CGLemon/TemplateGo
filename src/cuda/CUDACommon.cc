#ifdef USE_CUDA
#include "cuda/CUDACommon.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

void CublasError(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    // const char *s = CublasGetErrorString(status);
    std::cerr << "CUBLAS error: "
              << "\n"; //<< s << "\n";
    exit(-1);
  }
}

void CudaError(cudaError_t status) {
  if (status != cudaSuccess) {
    const char *s = cudaGetErrorString(status);
    std::cerr << "CUDA Error: " << s << "\n";
    exit(-1);
  }
}

#ifdef USE_CUDNN
void CudnnError(cudnnStatus_t status) {
  if (status != CUDNN_STATUS_SUCCESS) {
    const char *s = cudnnGetErrorString(status);
    std::cerr << "CUDA Error: " << s << "\n";
    exit(-1);
  }
}
#endif

int cuda_get_device() {
  int n = 0;
  cudaError_t status = cudaGetDevice(&n);
  ReportCUDAErrors(status);
  return n;
}

cublasHandle_t blas_handle() {
  static int init[16] = {0};
  static cublasHandle_t handle[16];
  int i = cuda_get_device();
  if (!init[i]) {
    cublasCreate(&handle[i]);
    init[i] = 1;
  }
  return handle[i];
}
#endif
