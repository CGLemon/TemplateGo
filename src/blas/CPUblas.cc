#include "blas/CPUblas.h"
#include <cmath>

#ifdef USE_EIGEN
// Eigen helpers
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using EigenVectorMap = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
#endif

void gemm_nn(int M, int N, int K, float ALPHA, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = ALPHA * A[i * lda + k];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void gemm_nt(int M, int N, int K, float ALPHA, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += ALPHA * A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
}

void gemm_tn(int M, int N, int K, float ALPHA, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = ALPHA * A[k * lda + i];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void gemm_tt(int M, int N, int K, float ALPHA, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      register float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += ALPHA * A[i + k * lda] * B[k + j * ldb];
      }
      C[i * ldc + j] += sum;
    }
  }
}

void blas_gemm(bool TA, bool TB, int M, int N, int K, float ALPHA,
               const float *A, int lda, const float *B, int ldb, float BETA,
               float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * ldc + j] *= BETA;
    }
  }
  if (!TA && !TB) {
    gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  } else if (TA && !TB) {
    gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  } else if (!TA && TB) {
    gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  } else if (TA && TB) {
    gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
  }
}

template <>
void Gemm<false, false>::blas_gemm(int M, int N, int K, float ALPHA,
                                   const float *A, int lda, const float *B,
                                   int ldb, float BETA, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * ldc + j] *= BETA;
    }
  }
  gemm_nn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

template <>
void Gemm<true, false>::blas_gemm(int M, int N, int K, float ALPHA,
                                  const float *A, int lda, const float *B,
                                  int ldb, float BETA, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * ldc + j] *= BETA;
    }
  }
  gemm_tn(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

template <>
void Gemm<false, true>::blas_gemm(int M, int N, int K, float ALPHA,
                                  const float *A, int lda, const float *B,
                                  int ldb, float BETA, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * ldc + j] *= BETA;
    }
  }
  gemm_nt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

template <>
void Gemm<true, true>::blas_gemm(int M, int N, int K, float ALPHA,
                                 const float *A, int lda, const float *B,
                                 int ldb, float BETA, float *C, int ldc) {
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      C[i * ldc + j] *= BETA;
    }
  }
  gemm_tt(M, N, K, ALPHA, A, lda, B, ldb, C, ldc);
}

void blas::dense(const int inputs, const int outputs,
                 const std::vector<float> &input,
                 const std::vector<float> &weights,
                 std::vector<float> &output) {
#ifndef USE_BLAS
  Gemm<false, true>::blas_gemm(1, outputs, inputs, 1, input.data(), inputs,
                               weights.data(), inputs, 1, output.data(),
                               outputs);
#else
#ifdef USE_OPENBLAS

  cblas_sgemv(CblasRowMajor, CblasNoTrans, outputs, inputs, 1.0f, &weights[0],
              inputs, &input[0], 1, 0.0f, &output[0], 1);

#endif
#ifdef USE_EIGEN
  EigenVectorMap<float> y(output.data(), outputs);
  y.noalias() =
      ConstEigenMatrixMap<float>(weights.data(), inputs, outputs).transpose() *
      ConstEigenVectorMap<float>(input.data(), inputs);

#endif
#endif
}

void blas::fixed_gemm(const int M, const int N, const int K, const float alpha,
                      const std::vector<float> &A, const int lda,
                      const std::vector<float> &B, const int ldb,
                      const float beta, std::vector<float> &C, const int ldc) {

#ifndef USE_BLAS
  Gemm<false, false>::blas_gemm(M, N, K, alpha, A.data(), lda, B.data(), ldb,
                                beta, C.data(), ldc);
#else
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, &A[0],
              lda, &B[0], ldb, beta, &C[0], ldc);

#endif
#ifdef USE_EIGEN
  auto C_mat = EigenMatrixMap<float>(C.data(), N, M);
  C_mat.noalias() = ConstEigenMatrixMap<float>(B.data(), N, K) *
                    ConstEigenMatrixMap<float>(A.data(), K, M);
#endif
#endif
}

void blas::winograd_gemm(const int set_U, const int set_V, const int set_M,
                         const int M, const int N, const int K,
                         const float alpha, const std::vector<float> &A,
                         const int lda, const std::vector<float> &B,
                         const int ldb, const float beta, std::vector<float> &C,
                         const int ldc) {

#ifndef USE_BLAS
  Gemm<true, false>::blas_gemm(M, N, K, alpha, A.data() + set_U, lda,
                               B.data() + set_V, ldb, beta, C.data() + set_M,
                               ldc);
#else
#ifdef USE_OPENBLAS

  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, M, N, K, alpha,
              &A[set_U], lda, &B[set_V], ldb, beta, &C[set_M], ldc);

#endif
#ifdef USE_EIGEN

  auto C_mat = EigenMatrixMap<float>(C.data() + set_M, N, M);
  C_mat.noalias() =
      ConstEigenMatrixMap<float>(B.data() + set_V, N, K) *
      ConstEigenMatrixMap<float>(A.data() + set_U, M, K).transpose();

#endif
#endif
}

/*
0.00001 0.00001 0.00001 0.00004 0.00005 0.00004 0.00003 0.00002 0.00001
0.00002 0.00002 0.00002 0.00010 0.00009 0.00005 0.00005 0.00003 0.00001
0.00002 0.00002 0.00009 0.00029 0.00016 0.00057 0.03722 0.00004 0.00004
0.00001 0.00001 0.00012 0.00283 0.00965 0.08812 0.02435 0.00010 0.00001
0.00001 0.00002 0.00005 0.01174 0.73678 0.05139 0.03018 0.00006 0.00002
0.00001 0.00002 0.00017 0.00105 0.00187 0.00095 0.00013 0.00002 0.00002
0.00001 0.00002 0.00031 0.00021 0.00021 0.00015 0.00004 0.00003 0.00001
0.00001 0.00002 0.00001 0.00001 0.00002 0.00002 0.00002 0.00002 0.00001
0.00001 0.00001 0.00001 0.00001 0.00002 0.00002 0.00001 0.00001 0.00001
pass : 0.00000
NN eval = 0.485828
*/
