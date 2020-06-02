#include "cuda/CUDAKernel.h"
#include "cuda/CUDACommon.h"

template <typename T>
__global__ void addVectors_kernel(T* c, T* a, T* b, int size, int asize, int bsize,
                                  bool relu, bool useTanh, bool useSigmoid) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i < size) {
		float aVal = 0;
		float bVal = 0;
		if (a) aVal = (float)(a[i % asize]);
		if (b) bVal = (float)(b[i % bsize]);

		float cVal = aVal + bVal;

		if (relu && (cVal < 0)) cVal = 0;

		if (useTanh) {
		  cVal = tanh(cVal);
		}

		if (useSigmoid) {
		  cVal = 1.0f / (1.0f + exp(-cVal));
		}

		c[i] = (T)cVal;
	}
}

// Adds two vectors (possibly of different sizes), also do optional relu
// activation.
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                bool relu, bool use_tanh, bool use_sigmoid) {

	const int kBlockSize = KBLOCKSIZE;
	int blocks = DivUp(size, kBlockSize);
	assert(asize == bsize);
	
	addVectors_kernel<<<blocks, kBlockSize>>>(c, a, b, size, asize, bsize, relu,
		                                    use_tanh, use_sigmoid);
	check_error(cudaGetLastError());
}


template void addVectors<float>(float* c, float* a, float* b, int size,
                                int asize, int bsize, bool relu, bool use_tanh, bool use_sigmoid);

