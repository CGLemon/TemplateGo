#include "Blas.h"
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

void gemm_nn(int M, int N, int K, float alpha, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = alpha * A[i * lda + k];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void gemm_nt(int M, int N, int K, float alpha, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += alpha * A[i * lda + k] * B[j * ldb + k];
      }
      C[i * ldc + j] += sum;
    }
  }
}

void gemm_tn(int M, int N, int K, float alpha, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int k = 0; k < K; ++k) {
      float A_PART = alpha * A[k * lda + i];
      for (int j = 0; j < N; ++j) {
        C[i * ldc + j] += A_PART * B[k * ldb + j];
      }
    }
  }
}

void gemm_tt(int M, int N, int K, float alpha, const float *A, int lda,
             const float *B, int ldb, float *C, int ldc) {
#pragma omp parallel for
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N; ++j) {
      register float sum = 0;
      for (int k = 0; k < K; ++k) {
        sum += alpha * A[i + k * lda] * B[k + j * ldb];
      }
      C[i * ldc + j] += sum;
    }
  }
}

#define INPUT_MULTI(M, N, beta)   \
  for (int i = 0; i < M; ++i) {   \
    for (int j = 0; j < N; ++j) { \
      C[i * ldc + j] *= beta;     \
    }                             \
  }



template <>
void Gemm<false, false>::apply(int M, int N, int K,
                               float alpha,
                               const float *A, int lda,
                               const float *B, int ldb,
                               float beta,
                               float *C, int ldc) {
  INPUT_MULTI(M, N, beta);
  gemm_nn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

template <>
void Gemm<true, false>::apply(int M, int N, int K,
                              float alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              float beta,
                              float *C, int ldc) {
  INPUT_MULTI(M, N, beta);
  gemm_tn(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

template <>
void Gemm<false, true>::apply(int M, int N, int K,
                              float alpha,
                              const float *A, int lda,
                              const float *B, int ldb,
                              float beta,
                              float *C, int ldc) {
  INPUT_MULTI(M, N, beta);
  gemm_nt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

template <>
void Gemm<true, true>::apply(int M, int N, int K,
                             float alpha,
                             const float *A, int lda,
                             const float *B, int ldb,
                             float beta,
                             float *C, int ldc) {
  INPUT_MULTI(M, N, beta);
  gemm_tt(M, N, K, alpha, A, lda, B, ldb, C, ldc);
}

void Blas::fixed_gemm(const int M, const int N, const int K,
                      const float alpha, 
                      const float *A, const int lda,
                      const float *B, const int ldb,
                      const float beta,
                      float *C, const int ldc) {

#ifndef USE_BLAS
  Gemm<false, false>::apply(M, N, K,
                            alpha,
                            A, lda,
                            B, ldb,
                            beta,
                            C, ldc);
#else
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
              M, N, K,
              alpha,
              A, lda,
              B, ldb,
              beta,
              C, ldc);

#endif
#ifdef USE_EIGEN
  auto C_mat = EigenMatrixMap<float>(C, N, M);
  C_mat.noalias() = 
      ConstEigenMatrixMap<float>(B, N, K) *
      ConstEigenMatrixMap<float>(A, K, M);
#endif
#endif
}


void Blas::winograd_gemm(const int set_U, const int set_V, const int set_M,
                         const int M, const int N, const int K,
                         const float alpha, 
                         const float *A, const int lda,
                         const float *B, const int ldb,
                         const float beta,
                         float *C, const int ldc) {

#ifndef USE_BLAS
  Gemm<true, false>::apply(M, N, K,
                           alpha,
                           A + set_U, lda,
                           B + set_V, ldb, 
                           beta, 
                           C + set_M, ldc);

#else
#ifdef USE_OPENBLAS
  cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
              M, N, K, 
              alpha,
              A + set_U, lda,
              B + set_V, ldb, 
              beta, 
              C + set_M, ldc);

#endif
#ifdef USE_EIGEN

  auto C_mat = EigenMatrixMap<float>(C + set_M, N, M);
  C_mat.noalias() =
      ConstEigenMatrixMap<float>(B + set_V, N, K) *
      ConstEigenMatrixMap<float>(A + set_U, M, K).transpose();

#endif
#endif
}



void Blas::dense(const int input_size,
                 const int output_size,
                 const int batch_size,
                 const float *inputs,
                 const float *kernel,
                 float *outputs) {

  static constexpr float alpha = 1.0f;
  static constexpr float beta = 0.0f;

#ifndef USE_BLAS
  Gemm<false, true>::apply(batch_size, output_size, input_size,
                          //    M            N            K
                           alpha,
                           inputs, input_size,
                           kernel, input_size,
                           beta,
                           outputs, output_size);
#else
#ifdef USE_OPENBLAS
  if (batch_size == 1) {
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                  // M     K
                  output_size, input_size, 1.0f, weights,
                  input_size, inputs, 1, 0.0f, outputs, 1);
  } else {
    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                output_size, batch_size,  input_size,         
                alpha,              
                kernel, input_size,
                inputs, input_size,
                beta, 
                outputs, output_size); 
  }

#endif
#ifdef USE_EIGEN
  if (batch_size == 1) {
    EigenVectorMap<float> y(outputs, output_size);
    y.noalias() =
        ConstEigenMatrixMap<float>(kernel, input_size, output_size).transpose() *
        ConstEigenVectorMap<float>(inputs, input_size);
  } else {
    auto C_mat = EigenMatrixMap<float>(outputs, output_size, batch_size);
    C_mat.noalias() =
        ConstEigenMatrixMap<float>(kernel, input_size, output_size)
            .transpose() *
        ConstEigenMatrixMap<float>(inputs, input_size, batch_size);

  }
#endif
#endif
}

void MatMulLayer::Forward(const size_t input_channels,
                          const size_t output_channels,
                          const std::vector<float> &input,
                          const std::vector<float> &weights,
                          std::vector<float> &output) {

  const int batch_size = 1;
  Blas::fixed_gemm(batch_size,
                   (int)output_channels,
                   (int)input_channels,
                   1.0f,
                   input.data(),
                   (int)input_channels,
                   weights.data(),
                   (int)output_channels,
                   0.0f,
                   output.data(),
                   (int)output_channels);

}


template<int conv_size>
void InflateGlobalPool<conv_size>::Forward(const size_t input_channels,
                                           const size_t output_channels,
                                           const std::vector<float> &input,
                                           std::vector<float> &output) {

  const size_t o_size = output.size();
  const size_t i_size = input.size();
  assert(o_size = 3 * i_size);

  const float *input_ptr = input.data();
  float *layer_1 = output.data();
  float *layer_2 = output.data() + i_size;
  float *layer_3 = output.data() + 2*i_size;

  for (auto i = size_t{0}; i < input_channels; ++i) {
    float Max = *input_ptr;
    float Sum = 0.0f;
    for (auto b = size_t{0}; b < spatial_size; ++b) {
      const float val = *input_ptr;
      Sum += val;
      if (Max < val) {
        Max = val;
      }
      input_ptr++;
    }

    const float Mean = Sum / spatial_size;
    for (auto b = size_t{0}; b < spatial_size; ++b) {

      *layer_1 = Mean;
      *layer_2 = Mean;
      *layer_3 = Max;

      layer_1++;
      layer_2++;
      layer_3++;
    }
  }
}


std::vector<float> Activation::Softmax(const std::vector<float> &input,
                                       const float temperature) {
  auto output = std::vector<float>{};
  output.reserve(input.size());

  const auto alpha = *std::max_element(cbegin(input), cend(input));
  auto denom = 0.0f;

  for (const auto in_val : input) {
    auto val = std::exp((in_val - alpha) / temperature);
    denom += val;
    output.push_back(val);
  }

  for (auto &out : output) {
    out /= denom;
  }

  return output;
}


std::vector<float> Activation::Tanh (const std::vector<float> &input) {

  auto output = std::vector<float>{};
  output.reserve(input.size());

  for (const auto &v : input) {
    output.push_back(std::tanh(v));
  }
  return output;
}

void FullyConnect::Forward(const int input_size,
                           const int output_size,
                           const std::vector<float> &input,
                           const std::vector<float> &weights,
                           const std::vector<float> &biases,
                           std::vector<float> &output, bool ReLU) {

  const auto lambda_ReLU = [](const auto val) {
    return (val > 0.0f) ? val : 0.0f;
  };

  const int batch = 1;
  Blas::dense(input_size,
              output_size,
              batch, 
              input.data(),
              weights.data(),
              output.data());


  for (auto o = size_t{0}; o < output_size; ++o) {
    auto val = biases[o] + output[o];
    if (ReLU) {
      val = lambda_ReLU(val);
    }
    output[o] = val;
  }
}

std::vector<float> FullyConnect::innerproduct(const int input_size,
                                              const int output_size,
                                              const std::vector<float> &input,
                                              const std::vector<float> &weights,
                                              const std::vector<float> &biases,
                                              bool ReLU) {

  auto output = std::vector<float>{};
  output.reserve(output_size);
  const auto lambda_ReLU = [](const auto val) {
    return (val > 0.0f) ? val : 0.0f;
  };

  const int batch = 1;
  Blas::dense(input_size,
              output_size,
              batch, 
              input.data(),
              weights.data(),
              output.data());
  
  for (auto o = size_t{0}; o < output_size; ++o) {
    auto val = biases[o] + output[o];
    if (ReLU) {
      val = lambda_ReLU(val);
    }
    output[o] = val;
  }
  return output;
}


