#ifndef CUDACOMMON_H_INCLUDE
#define CUDACOMMON_H_INCLUDE

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cudnn.h>

void CudnnError(cudnnStatus_t status, const char* file, const int& line);
void CublasError(cublasStatus_t status, const char* file, const int& line);
void CudaError(cudaError_t status, const char* file, const int& line);

#define ReportCUDNNErrors(status) CudnnError(status, __FILE__, __LINE__)
#define ReportCUBLASErrors(status) CublasError(status, __FILE__, __LINE__)
#define ReportCUDAErrors(status) CudaError(status, __FILE__, __LINE__)

inline int DivUp(int a, int b) { return (a + b - 1) / b; }


#endif
