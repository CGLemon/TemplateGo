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
    // char message[128];
    // sprintf(message, "CUBLAS error: %s (%s:%d) ",
    // CublasGetErrorString(status),
    //        file, line);
    // throw Exception(message);
  }
}

void CudaError(cudaError_t status) {
  if (status != cudaSuccess) {
    const char *s = cudaGetErrorString(status);
    std::cerr << "CUDA Error: " << s << "\n";
    exit(-1);
    // char message[128];
    // sprintf(message, "CUDA error: %s (%s:%d) ", cudaGetErrorString(status),
    //        file, line);
    // throw Exception(message);
  }
}

/*
void ReportCUDAErrors(cudaError_t status) {
  // cudaDeviceSynchronize();
  cudaError_t status2 = cudaGetLastError();
  if (status != cudaSuccess) {
    const char *s = cudaGetErrorString(status);
    std::cerr << "CUDA Error: " << s << "\n";
    exit(-1);
  }
  if (status2 != cudaSuccess) {
    const char *s = cudaGetErrorString(status);
    std::cerr << "CUDA Error Prev: " << s << "\n";
    exit(-1);
  }
}
*/

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
