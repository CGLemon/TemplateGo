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
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0;
            for (int k = 0; k < K; ++k) {
                sum += alpha * A[i + k * lda] * B[k + j * ldb];
            }
            C[i * ldc + j] += sum;
        }
    }
}

#define INPUT_MULTI(M, N, beta)       \
    for (int i = 0; i < M; ++i) {     \
        for (int j = 0; j < N; ++j) { \
            C[i * ldc + j] *= beta;   \
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

#undef INPUT_MULTI

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
                             alpha,
                             inputs, input_size,
                             kernel, input_size,
                             beta,
                             outputs, output_size);
#else
#ifdef USE_OPENBLAS
    if (batch_size == 1) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    output_size, input_size, 1.0f, kernel,
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

std::vector<float> Activation::Softmax(const std::vector<float> &input,
                                       const float temperature) {
    auto output = std::vector<float>{};
    output.reserve(input.size());

    const auto alpha = *std::max_element(std::begin(input), std::end(input));
    auto denom = 0.0f;

    for (const auto in_val : input) {
        auto val = std::exp((in_val - alpha) / temperature);
        denom += val;
        output.emplace_back(val);
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
        output.emplace_back(std::tanh(v));
    }
    return output;
}

std::vector<float> Activation::Sigmoid(const std::vector<float> &input) {

    const auto lambda_sigmoid = [](const auto val) {
        return 1.0f / (1.0f + std::exp(-val));
    };

    auto output = std::vector<float>{};
    output.reserve(input.size());

    for (const auto &v : input) {
        output.emplace_back(lambda_sigmoid(v));
    }

    return output;
}

void FullyConnect::Forward(const int input_size,
                           const int output_size,
                           const std::vector<float> &input,
                           const std::vector<float> &weights,
                           const std::vector<float> &biases,
                           std::vector<float> &output, bool ReLU) {

    const auto lambda_ReLU = [](const auto val) -> float {
        return (val > 0.0f) ? val : 0.0f;
    };

    static constexpr int batch = 1;
    Blas::dense(input_size,
                output_size,
                batch, 
                input.data(),
                weights.data(),
                output.data());

    if (ReLU) {
        for (auto o = int{0}; o < output_size; ++o) {
            output[o] = lambda_ReLU(biases[o] + output[o]);
        }
    } else {
        for (auto o = int{0}; o < output_size; ++o) {
            output[o] = biases[o] + output[o];
        }
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
    const auto lambda_ReLU = [](const auto val) -> float {
        return (val > 0.0f) ? val : 0.0f;
    };

    static constexpr int batch = 1;
    Blas::dense(input_size,
                output_size,
                batch, 
                input.data(),
                weights.data(),
                output.data());
  
    if (ReLU) {
        for (auto o = int{0}; o < output_size; ++o) {
            output[o] = lambda_ReLU(biases[o] + output[o]);
        }
    } else {
        for (auto o = int{0}; o < output_size; ++o) {
            output[o] = biases[o] + output[o];
        }
    }
    return output;
}
