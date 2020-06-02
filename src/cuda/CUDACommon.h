#ifndef CUDACOMMON_H_INCLUDE
#define CUDACOMMON_H_INCLUDE
#include <cudnn.h>
#include <cublas_v2.h>

#define BLOCK 512
void check_error(cudaError_t status);

#endif
