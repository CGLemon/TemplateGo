#ifndef BLAS_H_INCLUDE
#define BLAS_H_INCLUDE

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

#include "Winograd_helper.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>

template <bool TA, bool TB> 
class Gemm {
public:
    static void apply(int M, int N, int K,
                      float alpha,
                      const float *A, int lda,
                      const float *B, int ldb,
                      float beta,
                      float *C, int ldc);
};

// TODO: All merge in one blas
class Blas {
public:
    // For convolution
    static void fixed_gemm(const int M, const int N, const int K,
                           const float alpha, 
                           const float *A, const int lda,
                           const float *B, const int ldb,
                           const float beta,
                           float *C, const int ldc);

    // For Winograd
    static void winograd_gemm(const int set_U, const int set_V, const int set_M,
                              const int M, const int N, const int K,
                              const float alpha, 
                              const float *A, const int lda,
                              const float *B, const int ldb,
                              const float beta,
                              float *C, const int ldc);

    // For fullyconnect
    static void dense(const int inputs,
                      const int outputs,
                      const int batch_size,
                      const float *input,
                      const float *kernel,
                      float *output);

};

template<int CONV_SIZE>
class InputPool {
public:
    InputPool() = delete;
    static void Forward(const size_t input_size,
                        const size_t channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights_w,
                        const std::vector<float> &weights_b,
                        std::vector<float> &output);

private:
    static constexpr auto width = CONV_SIZE;
    static constexpr auto height = CONV_SIZE;
    static constexpr auto spatial_size = width * height;

};


template<int CONV_SIZE>
class GlobalAvgPool {
public:
    GlobalAvgPool() = delete;
    static void Forward(const size_t input_channels,
                        const std::vector<float> &input,
                        std::vector<float> &output);

private:
    static constexpr auto width = CONV_SIZE;
    static constexpr auto height = CONV_SIZE;
    static constexpr auto spatial_size = width * height;

};

template<int CONV_SIZE>
class SEUnit {
public:
    SEUnit() = delete;
    static void Forward(const size_t channels,
                        const size_t se_size,
                        std::vector<float> &input,
                        const std::vector<float> &residual,
                        const std::vector<float> &weights_w1,
                        const std::vector<float> &weights_b1,
                        const std::vector<float> &weights_w2,
                        const std::vector<float> &weights_b2);

private:
    static void SEProcess(const size_t channels,
                          std::vector<float> &input,
                          const std::vector<float> &residual,
                          const std::vector<float> &scale);

    static constexpr auto width = CONV_SIZE;
    static constexpr auto height = CONV_SIZE;
    static constexpr auto spatial_size = width * height;

};



class Activation {
public:
    static std::vector<float> Softmax(const std::vector<float> &input,
                                      const float temperature = 1.0f);

    static std::vector<float> Tanh(const std::vector<float> &input);

    static std::vector<float> Sigmoid(const std::vector<float> &input);
};


template<int CONV_SIZE>
class Convolve1 {
public:
    Convolve1() = delete;
    static void Forward(const size_t input_channels,
                        const size_t output_channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        std::vector<float> &output);

private:
    static constexpr auto width = CONV_SIZE;
    static constexpr auto height = CONV_SIZE;
    static constexpr auto spatial_size = width * height;
};


template<int CONV_SIZE>
class Batchnorm {
public:
    Batchnorm() = delete;
    static void Forward(const size_t channels,
                        std::vector<float> &input,
                        const std::vector<float> &means,
                        const std::vector<float> &stddevs,
                        const float *const eltwise = nullptr,
                        const bool ReLU = true);

private:
    static constexpr auto width = CONV_SIZE;
    static constexpr auto height = CONV_SIZE;
    static constexpr auto spatial_size = width * height;
};


template<int CONV_SIZE>
class winograd_convolve3 {
public:
    winograd_convolve3() = delete;
    static void Forward(const size_t input_channels,
                        const size_t output_channels,
                        const std::vector<float> &input,
                        const std::vector<float> &U,
                        std::vector<float> &V,
                        std::vector<float> &M,
                        std::vector<float> &output);

    static std::pair<size_t, size_t> get_workspace_size(const size_t input_channels,
                                                        const size_t output_channels);
private:
    static void transform_in(const std::vector<float> &in,
                             std::vector<float> &V,
                             const int C);

    static void sgemm(const std::vector<float> &U,
                      const std::vector<float> &V,
                      std::vector<float> &M,
                      const int C,
                      const int K);

    static void transform_out(const std::vector<float> &M,
                              std::vector<float> &Y,
                              const int K);
    static constexpr auto WINOGRAD_WTILES = (CONV_SIZE / WINOGRAD_M + (CONV_SIZE % WINOGRAD_M != 0));
    static constexpr auto WTILES = WINOGRAD_WTILES;
    static constexpr auto WINOGRAD_P = WINOGRAD_WTILES * WINOGRAD_WTILES;
    static constexpr auto W = CONV_SIZE;
    static constexpr auto H = CONV_SIZE;
    static constexpr auto filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;
};


template<int CONV_SIZE>
class Convolve {
public:
    Convolve() = delete;
    static void Forward(const size_t filter_size,
                        const size_t input_channels,
                        const size_t output_channels,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        std::vector<float> &col,
                        std::vector<float> &output);

    static size_t get_workspace_size(const size_t filter_size,
                                     const size_t input_channels);

private:
    static void im2col(const size_t filter_size,
                       const int channels,
                       const std::vector<float> &input,
                       std::vector<float> &col);

    static constexpr auto width = CONV_SIZE;
    static constexpr auto height = CONV_SIZE;
    static constexpr auto spatial_size = width * height;
};

class FullyConnect {
public:
    FullyConnect() = delete;
    static void Forward(const int inputs_size,
                        const int outputs_size,
                        const std::vector<float> &input,
                        const std::vector<float> &weights,
                        const std::vector<float> &biases,
                        std::vector<float> &output, bool ReLU);

    static std::vector<float> innerproduct(const int inputs_size,
                                           const int outputs_size,
                                           const std::vector<float> &input,
                                           const std::vector<float> &weights,
                                           const std::vector<float> &biases,
                                           bool ReLU);
};

template<int CONV_SIZE>
void winograd_convolve3<CONV_SIZE>::transform_in(const std::vector<float> &in,
                                                 std::vector<float> &V, const int C) {

    constexpr int P = WINOGRAD_P;
    constexpr int Wpad = 2 + WINOGRAD_M * WTILES;
    constexpr int buffersize = 32;

    std::array<std::array<float, Wpad>, Wpad> in_pad{0.0f};

    std::array<float, buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;
    int buffer_offset = 0;
    int buffer_entries = 0;

    // multiple vector [i0..i5] by Bt and produce [o0..o5]
    // const auto Bt = std::array<float, WINOGRAD_TILE>
    //           {1.0f,  0.0f,     -5.0f/2.0f,  0.0f,      1.0f, 0.0f,
    //            0.0f, -SQ2,      -2.0f,       SQ2/2.0f,  1.0f, 0.0f,
    //            0.0f,  SQ2,      -2.0f,      -SQ2/2.0f,  1.0f, 0.0f,
    //            0.0f, -SQ2/2.0f, -1.0f/2.0f,  SQ2,       1.0f, 0.0f,
    //            0.0f,  SQ2/2.0f, -1.0f/2.0f, -SQ2,       1.0f, 0.0f,
    //            0.0f,  1.0f,      0.0f,      -5.0f/2.0f, 0.0f, 1.0f};
    auto multiply_bt = [](float &o0, float &o1, float &o2, float &o3, float &o4,
                          float &o5, float i0, float i1, float i2, float i3,
                          float i4, float i5) {
        auto i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
        auto i4m2 = i2 * -2.0f + i4 * 1.0f;

        o0 = i0 + i2 * (-5.0f / 2.0f) + i4;
        o1 = i3m1 + i4m2;
        o2 = -i3m1 + i4m2;

        auto i3m1_2 = i3 * (SQ2) + i1 * (-SQ2 / 2.0f);
        auto i4m2_2 = i2 * (-1.0f / 2.0f) + i4;

        o3 = i3m1_2 + i4m2_2;
        o4 = -i3m1_2 + i4m2_2;

        o5 = i1 + i3 * (-5.0f / 2.0f) + i5;
    };

    for (auto ch = 0; ch < C; ch++) {
        for (auto yin = 0; yin < H; yin++) {
            for (auto xin = 0; xin < W; xin++) {
                in_pad[yin + 1][xin + 1] = in[ch * (W * H) + yin * W + xin];
            }
        }
        for (auto block_y = 0; block_y < WTILES; block_y++) {
            // Tiles overlap by 2
            const auto yin = WINOGRAD_M * block_y;
            for (auto block_x = 0; block_x < WTILES; block_x++) {
                const auto xin = WINOGRAD_M * block_x;
#define DECL_T1(XX)                                                            \
  float T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4,       \
      T1_##XX##_5;
                DECL_T1(0)
                DECL_T1(1)
                DECL_T1(2)
                DECL_T1(3)
                DECL_T1(4)
                DECL_T1(5)

              // Calculates transpose(B).x.B
#define MULTIPLY_BT(XX)                                                        \
  multiply_bt(T1_0_##XX, T1_1_##XX, T1_2_##XX, T1_3_##XX, T1_4_##XX,           \
              T1_5_##XX, in_pad[yin + 0][xin + XX], in_pad[yin + 1][xin + XX], \
              in_pad[yin + 2][xin + XX], in_pad[yin + 3][xin + XX],            \
              in_pad[yin + 4][xin + XX], in_pad[yin + 5][xin + XX]);
                MULTIPLY_BT(0)
                MULTIPLY_BT(1)
                MULTIPLY_BT(2)
                MULTIPLY_BT(3)
                MULTIPLY_BT(4)
                MULTIPLY_BT(5)

#define MULTIPLY_B(XX)                                                         \
  multiply_bt(buffer[buffersize * (XX * WINOGRAD_ALPHA + 0) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 1) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 2) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 3) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 4) + buffer_entries], \
              buffer[buffersize * (XX * WINOGRAD_ALPHA + 5) + buffer_entries], \
              T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4, \
              T1_##XX##_5);
                MULTIPLY_B(0)
                MULTIPLY_B(1)
                MULTIPLY_B(2)
                MULTIPLY_B(3)
                MULTIPLY_B(4)
                MULTIPLY_B(5)

                if (buffer_entries == 0) {
                    buffer_offset = ch * P + block_y * WTILES + block_x;
                }
                buffer_entries++;

                if (buffer_entries >= buffersize ||
                    (ch == C - 1 && block_x == WTILES - 1 && block_y == WTILES - 1)) {

                    for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                        for (auto entry = 0; entry < buffer_entries; entry++) {
                            V[i * C * P + buffer_offset + entry] =
                                buffer[i * buffersize + entry];
                        }
                    }
                    buffer_entries = 0;
                }
            }
        }
    }
}

template<int CONV_SIZE>
void winograd_convolve3<CONV_SIZE>::sgemm(const std::vector<float> &U,
                                          const std::vector<float> &V,
                                          std::vector<float> &M, const int C,
                                          const int K) {
    constexpr auto P = WINOGRAD_P;

    for (int b = 0; b < WINOGRAD_TILE; b++) {
        const int offset_u = b * K * C;
        const int offset_v = b * C * P;
        const int offset_m = b * K * P;

        Blas::winograd_gemm(offset_u,
                            offset_v,
                            offset_m,
                            K,
                            P,
                            C,
                            1.0f,
                            U.data(),
                            K,
                            V.data(),
                            P,
                            0.0f,
                            M.data(),
                            P);
    }
}

template<int CONV_SIZE>
void winograd_convolve3<CONV_SIZE>::transform_out(const std::vector<float> &M,
                                                  std::vector<float> &Y, const int K) {
    constexpr auto P = WINOGRAD_P;

    // multiple vector [i0..i5] by At and produce [o0..o3]
    // const auto At = std::array<float, WINOGRAD_ALPHA * WINOGRAD_M>
    //       {1.0f, 1.0f,      1.0f,       1.0f,      1.0f,     0.0f,
    //        0.0f, SQ2/2.0f, -SQ2/2.0f,   SQ2,      -SQ2,      0.0f,
    //        0.0f, 1.0f/2.0f, 1.0f/2.0f,  2.0f,      2.0f,     0.0f,
    //        0.0f, SQ2/4.0f, -SQ2/4.0f,   2.0f*SQ2, -2.0f*SQ2, 1.0f};
    auto multiply_at = [](float &o0, float &o1, float &o2, float &o3, float i0,
                          float i1, float i2, float i3, float i4, float i5) {
        auto t1p2 = (i1 + i2) * (1.0f / 2.0f);
        auto t1m2 = (i1 - i2) * (SQ2 / 4.0f);
        auto t3p4 = i3 + i4;
        auto t3m4 = (i3 - i4) * (SQ2);

        o0 = i0 + t1p2 + t1p2 + t3p4;
        o1 = t1m2 + t1m2 + t3m4;
        o2 = t1p2 + t3p4 + t3p4;
        o3 = t1m2 + t3m4 + t3m4 + i5;
    };

    for (auto k = 0; k < K; k++) {
        for (auto block_x = 0; block_x < WTILES; block_x++) {
            const auto x = WINOGRAD_M * block_x;
            for (auto block_y = 0; block_y < WTILES; block_y++) {
                const auto y = WINOGRAD_M * block_y;

                const auto b = block_y * WTILES + block_x;
                using WinogradTile =
                    std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_ALPHA>;

                auto temp_m = WinogradTile{};
                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        temp_m[xi][nu] = M[(xi * WINOGRAD_ALPHA + nu) * K * P + k * P + b];
                    }
                }
                auto temp = std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_M>{};
                auto o = std::array<std::array<float, WINOGRAD_M>, WINOGRAD_M>{};

                // Calculates transpose(A).temp_m.A
                for (auto j = 0; j < WINOGRAD_ALPHA; j++) {
                    multiply_at(temp[0][j], temp[1][j], temp[2][j], temp[3][j],
                                temp_m[0][j], temp_m[1][j], temp_m[2][j], temp_m[3][j],
                                temp_m[4][j], temp_m[5][j]);
                }

                for (auto i = 0; i < WINOGRAD_M; i++) {
                    multiply_at(o[i][0], o[i][1], o[i][2], o[i][3], temp[i][0],
                                temp[i][1], temp[i][2], temp[i][3], temp[i][4],
                                temp[i][5]);
                }

                const auto y_ind = k * H * W + y * W + x;
                for (auto i = 0; i < WINOGRAD_M; i++) {
                    for (auto j = 0; j < WINOGRAD_M; j++) {
                        if (y + i < H && x + j < W) {
                            Y[y_ind + i * W + j] = o[i][j];
                        }
                    }
                }
            }
        }
    }
}

template<int CONV_SIZE>
void winograd_convolve3<CONV_SIZE>::Forward(const size_t input_channels,
                                            const size_t output_channels,
                                            const std::vector<float> &input,
                                            const std::vector<float> &U,
                                            std::vector<float> &V,
                                            std::vector<float> &M,
                                            std::vector<float> &output) {

    transform_in(input, V, input_channels);
    sgemm(U, V, M, input_channels, output_channels);
    transform_out(M, output, output_channels);
}


template<int CONV_SIZE>
std::pair<size_t, size_t> winograd_convolve3<CONV_SIZE>::get_workspace_size(const size_t input_channels,
                                                                            const size_t output_channels) {

    auto winograd_V_size = WINOGRAD_TILE * input_channels * WINOGRAD_P;
    auto winograd_M_size = WINOGRAD_TILE * output_channels * WINOGRAD_P;
    return std::make_pair(winograd_V_size, winograd_M_size);
}

template<int CONV_SIZE>
void Convolve1<CONV_SIZE>::Forward(const size_t input_channels,
                                   const size_t output_channels,
                                   const std::vector<float> &input,
                                   const std::vector<float> &weights,
                                   std::vector<float> &output) {

     Blas::fixed_gemm((int)output_channels,
                       spatial_size,
                       (int)input_channels,
                       1.0f,
                       weights.data(),
                       (int)input_channels,
                       input.data(),
                       spatial_size,
                       0.0f,
                       output.data(),
                       spatial_size);
}

template<int CONV_SIZE>
void Batchnorm<CONV_SIZE>::Forward(const size_t channels,
                                   std::vector<float> &input,
                                   const std::vector<float> &means,
                                   const std::vector<float> &stddevs,
                                   const float *const eltwise,
                                   const bool ReLU) {

    const auto lambda_ReLU = [&](const auto val) {
        return (val > 0.0f || (!ReLU)) ? val : 0.0f;
    };

    float *input_ptr = input.data();
    const float *res = eltwise;
    if (eltwise) {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto mean = means[c];
            const auto scale_stddev = stddevs[c];

            for (auto b = size_t{0}; b < spatial_size; b++) {
                float value = *input_ptr;
                value = scale_stddev * (value - mean) + *res;
                *input_ptr = lambda_ReLU(value);

                input_ptr++;
                res++;
            }
        }
    } else {
        for (auto c = size_t{0}; c < channels; ++c) {
            const auto mean = means[c];
            const auto scale_stddev = stddevs[c];

            for (auto b = size_t{0}; b < spatial_size; b++) {
                float value = *input_ptr;
                value = scale_stddev * (value - mean);
                *input_ptr = lambda_ReLU(value);
                input_ptr++;
            }
        }
    }
}


template<int CONV_SIZE>
void Convolve<CONV_SIZE>::im2col(const size_t filter_size,
                                 const int channels,
                                 const std::vector<float> &input,
                                 std::vector<float> &output) {

  int pad = (filter_size / 2);
  unsigned int output_h = height + 2 * pad - filter_size + 1;
  unsigned int output_w = width + 2 * pad - filter_size + 1;

  const float *data_im = input.data();
  float *data_col = output.data();

    for (int channel = channels; channel--; data_im += spatial_size) {
        for (unsigned int kernel_row = 0; kernel_row < filter_size; kernel_row++) {
            for (unsigned int kernel_col = 0; kernel_col < filter_size;  kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (unsigned(input_row) < height) {
                        int input_col = -pad + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (unsigned(input_col) < width) {
                                *(data_col++) = data_im[input_row * width + input_col];
                            } else {
                                *(data_col++) = 0;
                            }
                            input_col++;
                        }
                    } else {
                        for (int output_cols = output_w; output_cols; output_cols--) {
                            *(data_col++) = 0;
                        }
                    }
                    input_row++;
                }
            }
        }
    }
}


template<int CONV_SIZE>
void Convolve<CONV_SIZE>::Forward(const size_t filter_size,
                                  const size_t input_channels,
                                  const size_t output_channels,
                                  const std::vector<float> &input,
                                  const std::vector<float> &weights,
                                  std::vector<float> &col,
                                  std::vector<float> &output) {
    const int filter_len = filter_size * filter_size;
    const int filter_dim = filter_len * input_channels;
    assert(output_channels * spatial_size == output.size());

    // Weight shape (output, input, filter_size, filter_size)
    // 96 18 3 3
    // C←αAB + βC
    // outputs[96,19x19] = weights[96,18x3x3] x col[18x3x3,19x19]
    // M Number of rows in matrices A and C.
    // N Number of columns in matrices B and C.
    // K Number of columns in matrix A; number of rows in matrix B.
    // lda The size of the first dimention of matrix A; if you are
    // passing a matrix A[m][n], the value should be m.
    //    cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B,
    //                ldb, beta, C, N);

    im2col(filter_size, input_channels, input, col);

    Blas::fixed_gemm((int)output_channels,
                      spatial_size,
                      filter_dim,
                      1.0f,
                      weights.data(),
                      filter_dim,
                      col.data(),
                      spatial_size,
                      0.0f,
                      output.data(),
                      spatial_size);

}

template<int CONV_SIZE>
size_t Convolve<CONV_SIZE>::get_workspace_size(const size_t filter_size,
                                               const size_t input_channels) {

    const size_t filter_len = filter_size * filter_size;
    const size_t filter_dim = filter_len * input_channels;
    return filter_dim * width * height;
}

template<int CONV_SIZE>
void GlobalAvgPool<CONV_SIZE>::Forward(const size_t channels,
                                    const std::vector<float> &input,
                                    std::vector<float> &output) {

    const float *input_ptr = input.data();

    for (auto c = size_t{0}; c < channels; ++c) {
        float Sum = 0.0f;
        for (auto b = size_t{0}; b < spatial_size; ++b) {
            Sum += *input_ptr;
            input_ptr++;
        }

        const float Mean = Sum / (float)spatial_size;
        output[c] = Mean;
    }
}

template<int CONV_SIZE>
void SEUnit<CONV_SIZE>::Forward(const size_t channels,
                                const size_t se_size,
                                std::vector<float> &input,
                                const std::vector<float> &residual,
                                const std::vector<float> &weights_w1,
                                const std::vector<float> &weights_b1,
                                const std::vector<float> &weights_w2,
                                const std::vector<float> &weights_b2) {

    using pooling = GlobalAvgPool<CONV_SIZE>;
    auto pool = std::vector<float>(2 * channels);
    auto fc_out = std::vector<float>(se_size);

    pooling::Forward(channels, input, pool);
    FullyConnect::Forward(channels, se_size, pool, weights_w1, weights_b1, fc_out, true);
    FullyConnect::Forward(se_size, 2*channels, fc_out, weights_w2, weights_b2, pool, false);

    SEProcess(channels, input, residual, pool);
}

template<int CONV_SIZE>
void SEUnit<CONV_SIZE>::SEProcess(const size_t channels,
                                  std::vector<float> &input,
                                  const std::vector<float> &residual,
                                  const std::vector<float> &scale) {

    const auto lambda_ReLU = [](const auto val) {
        return (val > 0.0f) ? val : 0;
    };

    const auto lambda_sigmoid = [](const auto val) {
        return 1.0f / (1.0f + std::exp(-val));
    };

    auto gamma_ptr = scale.data();
    auto beta_ptr = scale.data() + channels;
    auto input_ptr = input.data();
    auto res_ptr = residual.data();


    for (auto c = size_t{0}; c < channels; ++c) {
        const auto gamma = lambda_sigmoid(*gamma_ptr);
        const auto beta = *beta_ptr;

        gamma_ptr++;
        beta_ptr++;

        for (auto i = size_t{0}; i < spatial_size; ++i) {
            float value = *input_ptr;
            *input_ptr = lambda_ReLU(gamma * value + beta + *res_ptr);
            input_ptr++;
            res_ptr++;
        }
    }
}

template<int CONV_SIZE>
void InputPool<CONV_SIZE>::Forward(const size_t input_size,
                                   const size_t channels,
                                   const std::vector<float> &input,
                                   const std::vector<float> &weights_w,
                                   const std::vector<float> &weights_b,
                                   std::vector<float> &output) {

    auto fc_out = std::vector<float>(channels);
    FullyConnect::Forward(input_size, channels,
          input, weights_w, weights_b, fc_out, false);


    const auto lambda_ReLU = [](const auto val) {
        return (val > 0.0f) ? val : 0;
    };

    auto output_ptr = output.data();

    for (auto c = size_t{0}; c < channels; ++c) {
        float bais = fc_out[c];
        for (auto i = size_t{0}; i < spatial_size; ++i) {
            float value = *output_ptr;
            *output_ptr = lambda_ReLU(value + bais);
            output_ptr++;
        }
    }
}
#endif
