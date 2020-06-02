#ifndef CPUBLAS_H_INCLUDE
#define CPUBLAS_H_INCLUDE

#ifdef __APPLE__
#include <Accelerate/Accelerate.h>
#endif

#ifdef USE_MKL
#include <mkl.h>
#endif

#ifdef USE_OPENBLAS
#include <cblas.h>
#endif

#ifdef USE_EIGEN
#include <Eigen/Dense>
#endif

#include <vector>

class blas {
public:
	static void	fixed_gemm(const int M, const int N, const int K,
	                       const float alpha, 
	                       const std::vector<float>& A, const int lda,
	                       const std::vector<float>& B, const int ldb,
	                       const float beta,
	                       std::vector<float>& C, const int ldc);

	static void	winograd_gemm(const int set_U, const int set_V, const int set_M,
	                          const int M, const int N, const int K,
	                          const float alpha, 
	                          const std::vector<float>& A, const int lda,
	                          const std::vector<float>& B, const int ldb,
	                          const float beta,
	                          std::vector<float>& C, const int ldc);



	static void	dense(const int inputs,
		              const int outputs,
		              const std::vector<float>& input,
		              const std::vector<float>& weights,
		              std::vector<float>& output);



};




#endif
