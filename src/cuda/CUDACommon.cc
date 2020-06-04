
#ifdef USE_CUDA
#include "cuda/CUDACommon.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>

void ReportCUDAErrors(cudaError_t status) {
    //cudaDeviceSynchronize();
	cudaError_t status2 = cudaGetLastError();
	if (status != cudaSuccess)
	{   
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		std::cerr<<"CUDA Error: "<<s<<"\n"; 
		exit(-1);
	} 
	if (status2 != cudaSuccess)
	{   
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		std::cerr<<"CUDA Error Prev: "<<s<<"\n"; 
		exit(-1);
	} 
}
#endif


