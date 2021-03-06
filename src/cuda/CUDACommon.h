#ifndef CUDACOMMON_H_INCLUDE
#define CUDACOMMON_H_INCLUDE
#ifdef USE_CUDA


#include "Utils.h"
#include <cstdio>
#include <cuda_runtime.h>
//#include <cuda_fp16.h>
#include <cublas_v2.h>

#ifdef USE_CUDNN
#include <cudnn.h>
#endif

#define KBLOCKSIZE 256

#ifdef USE_CUDNN
void CudnnError(cudnnStatus_t status);
#define ReportCUDNNErrors(status) CudnnError(status)
cudnnHandle_t cudnn_handle();
#endif
void CublasError(cublasStatus_t status);
void CudaError(cudaError_t status);


#define ReportCUBLASErrors(status) CublasError(status)
#define ReportCUDAErrors(status) CudaError(status)

cublasHandle_t blas_handle();

int cuda_get_device();

inline static int DivUp(int a, int b) { return (a + b - 1) / b; }

struct CudaHandel {
#ifdef USE_CUDNN
    cudnnHandle_t cudnn_handel;
#endif
    cublasHandle_t cublas_handel;

    void apply();
};

void cuda_gpu_info();

#endif
#endif
