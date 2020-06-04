#ifndef CUDACOMMON_H_INCLUDE
#define CUDACOMMON_H_INCLUDE
#ifdef USE_CUDA
#include <cudnn.h>
//#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cstdio>

#define KBLOCKSIZE 512

void ReportCUDAErrors(cudaError_t status);

inline static int DivUp(int a, int b) { return (a + b - 1) / b; }

#endif
#endif
