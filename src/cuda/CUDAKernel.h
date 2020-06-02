#ifndef CUDABLAS_H_INCLUDE
#define CUDABLAS_H_INCLUDE
#include <cassert>

template <typename T>
void addVectors(T* c, T* a, T* b, int size, int asize, int bsize,
                bool relu, bool use_tanh, bool use_sigmoid);



#endif

