#include "cuda/CUDACommon.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <iostream>
void check_error(cudaError_t status)
{
    //cudaDeviceSynchronize();
	cudaError_t status2 = cudaGetLastError();
	if (status != cudaSuccess)
	{   
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		//printf("CUDA Error: %s\n", s);
		std::cerr<<"CUDA Error: "<<s<<"\n"; 
		exit(-1);
		//snprintf(buffer, 256, "CUDA Error: %s", s);
		//error(buffer);
	} 
	if (status2 != cudaSuccess)
	{   
		const char *s = cudaGetErrorString(status);
		char buffer[256];
		//printf("CUDA Error Prev: %s\n", s);
		std::cerr<<"CUDA Error Prev: "<<s<<"\n"; 
		exit(-1);
		//snprintf(buffer, 256, "CUDA Error Prev: %s", s);
		//error(buffer);
	} 
}
