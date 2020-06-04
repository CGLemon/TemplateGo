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
	
	addVectors_kernel<<<blocks, kBlockSize>>>(c, a, b, size, asize, bsize, relu,
    	                                      use_tanh, use_sigmoid);
	ReportCUDAErrors(cudaGetLastError());
}



template <typename T>
__global__ void addBias_NCHW_kernel(T* c, T* a, T* b,
                                    int N, int C, int H, int W) {
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	int size = N * C * H * W;
	if (i < size) {
		float aVal = (float)a[i];

	// All this math can be optimized, but the kernel is memory bound anyway.
		int biasIndex = (i / (H * W)) % C;
		float bVal = (float)b[biasIndex];

		float cVal = aVal + bVal;
		c[i] = (T)cVal;
	}
}

// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W) {
	int size = N * C * H * W;
	const int kBlockSize = 256;
	int blocks = DivUp(size, kBlockSize);

	addBias_NCHW_kernel<<<blocks, kBlockSize>>>(c, a, b, N, C, H, W);
	ReportCUDAErrors(cudaGetLastError());
}

template <typename T>
__global__ void batchNorm_kernel(T* output, const T* input, const T* skipInput,
                                 int N, int C, int H, int W, const float* means,
                                 const float* varMultipliers, bool relu) {
	int index = threadIdx.x + blockDim.x * blockIdx.x;

	int wIndex = 0;
	if (sizeof(T) == sizeof(float))
		wIndex = (index / (H * W)) % C;  // NCHW for fp32.
	else
		wIndex = index % C;  // NHWC for fp16.

	float el = input[index];
	float mean = means[wIndex];
	float varMulti = varMultipliers[wIndex];

	el -= mean;
	el *= varMulti;

	if (skipInput) el += (float)skipInput[index];

	if (relu && (el < 0)) el = 0;

	output[index] = (T)el;
}

// Every thread processes single element.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers, bool relu) {
	const int total_elements = N * C * H * W;
	const int kBlockSize = 256;
	int blocks = DivUp(total_elements, kBlockSize);

	batchNorm_kernel<<<blocks, kBlockSize>>>(output, input, skipInput, N, C, H, W,
    	                                     means, var_multipliers, relu);

	ReportCUDAErrors(cudaGetLastError());
}

template void addVectors<float>(float* c, float* a, float* b, int size,
                                int asize, int bsize, bool relu, bool use_tanh, bool use_sigmoid);

template void addBias_NCHW(float* c, float* a, float* b, int N, int C, int H, int W);

template void batchNorm(float* output, const float* input, const float* skipInput, 
                        int N, int C,int H, int W, float* means, float* var_multipliers, bool relu);

