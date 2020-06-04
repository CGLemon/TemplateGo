#ifndef CUDABLAS_H_INCLUDE
#define CUDABLAS_H_INCLUDE
#include <cassert>


// Adds two vectors (possibly of different sizes), also do optional
// activation (relu, tanh or sigmoid).
template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                bool relu, bool use_tanh, bool use_sigmoid);


// Add bias to convolution's output.
template <typename T>
void addBias_NCHW(T* c, T* a, T* b, int N, int C, int H, int W);


// Perform batch normilization.
template <typename T>
void batchNorm(T* output, const T* input, const T* skipInput, int N, int C,
               int H, int W, float* means, float* var_multipliers, bool relu);


#endif

