#include "CUDAbackend.h"

/*
template <typename T>
__global__ void policyMap_kernel(T* output, const T* input,
                             const short* indices, int N, int inputSize,
                             int usedSize, int outputSize) {
	  int tid = blockIdx.x * blockDim.x + threadIdx.x;

	  int n = tid / usedSize;
	  int i = tid % usedSize;

	  if (n >= N) return;

	  int j = indices[i];

	  if (j >= 0) {
			output[n * outputSize + j] = input[n * inputSize + i];
	  }
}
*/
