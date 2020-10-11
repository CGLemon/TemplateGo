#ifdef USE_CUDA

#include "cuda/CUDACommon.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>


void CublasError(cublasStatus_t status) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        // const char *s = CublasGetErrorString(status);
        std::cerr << "CUBLAS error: " << "\n"; //<< s << "\n";
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

cudnnHandle_t cudnn_handle() {
    static int init[16] = {0};
    static cudnnHandle_t handle[16];
    int i = cuda_get_device();
    if(!init[i]) {
        cudnnCreate(&handle[i]);
        init[i] = 1;
    }
    return handle[i];
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


void CudaHandel::apply() {
#ifdef USE_CUDNN
  cudnn_handel = cudnn_handle();
#endif
  cublas_handel = blas_handle();
}

void OutputSpec(const cudaDeviceProp & sDevProp) {
    Utils::auto_printf(" Device name: %s\n", sDevProp.name);
    Utils::auto_printf(" Device memory(MiB): %zu\n", (sDevProp.totalGlobalMem/(1024*1024)));
    Utils::auto_printf(" Memory per-block(KiB): %zu\n", (sDevProp.sharedMemPerBlock/1024));
    Utils::auto_printf(" Register per-block(KiB): %zu\n", (sDevProp.regsPerBlock/1024));
    Utils::auto_printf(" Warp size: %zu\n", sDevProp.warpSize);
    Utils::auto_printf(" Memory pitch(MiB): %zu\n", (sDevProp.memPitch/(1024*1024)));
    Utils::auto_printf(" Constant Memory(KiB): %zu\n", (sDevProp.totalConstMem/1024));
    Utils::auto_printf(" Max thread per-block: %zu\n", sDevProp.maxThreadsPerBlock);
    Utils::auto_printf(" Max thread dim: (%zu, %zu, %zu)\n", sDevProp.maxThreadsDim[0], sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2]);
    // Utils::auto_printf("Max grid size: (%zu, %zu, %zu)\n", sDevProp.maxGridSize[0], sDevProp.maxGridSize[1], sDevProp.maxGridSize[2]);
    Utils::auto_printf(" Ver: %zu.%zu\n", sDevProp.major, sDevProp.minor);
    Utils::auto_printf(" Clock: %zu(kHz)\n", (sDevProp.clockRate/1000));
    Utils::auto_printf(" textureAlignment: %zu\n", sDevProp.textureAlignment);
}

void cuda_gpu_info() {
    int iDeviceCount = 0;
    cudaGetDeviceCount(&iDeviceCount);
    Utils::auto_printf("Number of CUDA devices: %zu\n", iDeviceCount);

    if(iDeviceCount == 0) {
        Utils::auto_printf("No CUDA device\n");
        exit(-1);
    }

    for(int i = 0; i < iDeviceCount; ++i) {
        Utils::auto_printf("\n=== Device %zu ===\n", i);
        cudaDeviceProp sDeviceProp;
        cudaGetDeviceProperties(&sDeviceProp, i);
        OutputSpec(sDeviceProp);
    }
    Utils::auto_printf("\n");
    // cudaSetDevice(0);
}

#endif
