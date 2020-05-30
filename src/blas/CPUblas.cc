#include "blas/CPUblas.h"
#include "NetPipe.h"

#include <cmath>

std::vector<float> winograd_transform_f(const std::vector<float>& f,
                                        const int outputs,
                                        const int channels) {
    // F(4x4, 3x3) Winograd filter transformation
    // transpose(G.dot(f).dot(G.transpose()))
    // U matrix is transposed for better memory layout in SGEMM
    auto U = std::vector<float>(WINOGRAD_TILE * outputs * channels);
    const auto G = std::array<float, 3 * WINOGRAD_ALPHA>
                    { 1.0f,        0.0f,      0.0f,
                      -2.0f/3.0f, -SQ2/3.0f, -1.0f/3.0f,
                      -2.0f/3.0f,  SQ2/3.0f, -1.0f/3.0f,
                      1.0f/6.0f,   SQ2/6.0f,  1.0f/3.0f,
                      1.0f/6.0f,  -SQ2/6.0f,  1.0f/3.0f,
                      0.0f,        0.0f,      1.0f};

    auto temp = std::array<float, 3 * WINOGRAD_ALPHA>{};

    constexpr auto max_buffersize = 8;
    auto buffersize = max_buffersize;

    if (outputs % buffersize != 0) {
        buffersize = 1;
    }

    std::array<float, max_buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;

    for (auto c = 0; c < channels; c++) {
        for (auto o_b = 0; o_b < outputs/buffersize; o_b++) {
            for (auto bufferline = 0; bufferline < buffersize; bufferline++) {
                const auto o = o_b * buffersize + bufferline;

                for (auto i = 0; i < WINOGRAD_ALPHA; i++) {
                    for (auto j = 0; j < 3; j++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += G[i*3 + k] * f[o*channels*9 + c*9 + k*3 + j];
                        }
                        temp[i*3 + j] = acc;
                    }
                }

                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        auto acc = 0.0f;
                        for (auto k = 0; k < 3; k++) {
                            acc += temp[xi*3 + k] * G[nu*3 + k];
                        }
                        buffer[(xi * WINOGRAD_ALPHA + nu) * buffersize + bufferline] = acc;
                    }
                }
            }
            for (auto i = 0; i < WINOGRAD_ALPHA * WINOGRAD_ALPHA; i++) {
                for (auto entry = 0; entry < buffersize; entry++) {
                    const auto o = o_b * buffersize + entry;
                    U[i * outputs * channels
                      + c * outputs
                      + o] =
                    buffer[buffersize * i + entry];
                }
            }
        }
    }

    return U;
}


void winograd_transform_in(const std::vector<float>& in,
                           std::vector<float>& V,
                           const int C) {
    constexpr auto W = CONV2D_SIZE;
    constexpr auto H = CONV2D_SIZE;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;

    constexpr auto Wpad = 2 + WINOGRAD_M * WTILES;

    constexpr auto buffersize = 32;

    std::array<std::array<float, Wpad>, Wpad> in_pad{0.0f};

    std::array<float, buffersize * WINOGRAD_ALPHA * WINOGRAD_ALPHA> buffer;
    auto buffer_offset = 0;
    auto buffer_entries = 0;


    // multiple vector [i0..i5] by Bt and produce [o0..o5]
    // const auto Bt = std::array<float, WINOGRAD_TILE>
    //           {1.0f,  0.0f,     -5.0f/2.0f,  0.0f,      1.0f, 0.0f,
    //            0.0f, -SQ2,      -2.0f,       SQ2/2.0f,  1.0f, 0.0f,
    //            0.0f,  SQ2,      -2.0f,      -SQ2/2.0f,  1.0f, 0.0f,
    //            0.0f, -SQ2/2.0f, -1.0f/2.0f,  SQ2,       1.0f, 0.0f,
    //            0.0f,  SQ2/2.0f, -1.0f/2.0f, -SQ2,       1.0f, 0.0f,
    //            0.0f,  1.0f,      0.0f,      -5.0f/2.0f, 0.0f, 1.0f};
    auto multiply_bt = [](
        float & o0, float & o1, float & o2, float & o3, float & o4, float & o5,
        float i0, float i1, float i2, float i3, float i4, float i5
    ) {
        auto i3m1 = i1 * -SQ2 + i3 * (SQ2 / 2.0f);
        auto i4m2 = i2 * -2.0f + i4 * 1.0f;

        o0 = i0 + i2 * (-5.0f/2.0f) + i4;
        o1 = i3m1 + i4m2;
        o2 = -i3m1 + i4m2;

        auto i3m1_2 = i3 * (SQ2) + i1 * (-SQ2/2.0f);
        auto i4m2_2 = i2 * (-1.0f/2.0f) + i4;

        o3 = i3m1_2 + i4m2_2;
        o4 = -i3m1_2 + i4m2_2;

        o5 = i1 + i3 * (-5.0f/2.0f) + i5;
    };

    for (auto ch = 0; ch < C; ch++) {
        for (auto yin = 0; yin < H; yin++) {
            for (auto xin = 0; xin < W; xin++) {
                in_pad[yin + 1][xin + 1] = in[ch*(W*H) + yin*W + xin];
            }
        }
        for (auto block_y = 0; block_y < WTILES; block_y++) {
            // Tiles overlap by 2
            const auto yin = WINOGRAD_M * block_y;
            for (auto block_x = 0; block_x < WTILES; block_x++) {
                const auto xin = WINOGRAD_M * block_x;
#define DECL_T1(XX) \
                float T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4, T1_##XX##_5;
                DECL_T1(0)
                DECL_T1(1)
                DECL_T1(2)
                DECL_T1(3)
                DECL_T1(4)
                DECL_T1(5)

                // Calculates transpose(B).x.B
#define MULTIPLY_BT(XX) \
                multiply_bt( \
                    T1_0_##XX, T1_1_##XX, T1_2_##XX, T1_3_##XX, T1_4_##XX, T1_5_##XX, \
                    in_pad[yin + 0][xin + XX], \
                    in_pad[yin + 1][xin + XX], \
                    in_pad[yin + 2][xin + XX], \
                    in_pad[yin + 3][xin + XX], \
                    in_pad[yin + 4][xin + XX], \
                    in_pad[yin + 5][xin + XX] \
                );
                MULTIPLY_BT(0)
                MULTIPLY_BT(1)
                MULTIPLY_BT(2)
                MULTIPLY_BT(3)
                MULTIPLY_BT(4)
                MULTIPLY_BT(5)

#define MULTIPLY_B(XX) \
                multiply_bt( \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 0) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 1) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 2) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 3) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 4) + buffer_entries], \
                    buffer[buffersize * (XX * WINOGRAD_ALPHA + 5) + buffer_entries], \
                    T1_##XX##_0, T1_##XX##_1, T1_##XX##_2, T1_##XX##_3, T1_##XX##_4, T1_##XX##_5 \
                );
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
                            V[i*C*P + buffer_offset + entry] = buffer[i*buffersize + entry];
                        }
                    }
                    buffer_entries = 0;
                }
            }
        }
    }
}

void winograd_sgemm(const std::vector<float>& U,
                    const std::vector<float>& V,
                    std::vector<float>& M,
                    const int C, const int K) {
    constexpr auto P = WINOGRAD_P;

    for (auto b = 0; b < WINOGRAD_TILE; b++) {
        const auto offset_u = b * K * C;
        const auto offset_v = b * C * P;
        const auto offset_m = b * K * P;
#ifdef USE_BLAS
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans,
                    K, P, C,
                    1.0f,
                    &U[offset_u], K,
                    &V[offset_v], P,
                    0.0f,
                    &M[offset_m], P);
#else
        auto C_mat = EigenMatrixMap<float>(M.data() + offset_m, P, K);
        C_mat.noalias() =
           ConstEigenMatrixMap<float>(V.data() + offset_v, P, C)
            * ConstEigenMatrixMap<float>(U.data() + offset_u, K, C).transpose();
#endif
    }
}

void winograd_transform_out(const std::vector<float>& M,
                                     std::vector<float>& Y,
                                     const int K) {
    constexpr auto W = CONV2D_SIZE;
    constexpr auto H = CONV2D_SIZE;
    constexpr auto WTILES = WINOGRAD_WTILES;
    constexpr auto P = WINOGRAD_P;

    // multiple vector [i0..i5] by At and produce [o0..o3]
    // const auto At = std::array<float, WINOGRAD_ALPHA * WINOGRAD_M>
    //       {1.0f, 1.0f,      1.0f,       1.0f,      1.0f,     0.0f,
    //        0.0f, SQ2/2.0f, -SQ2/2.0f,   SQ2,      -SQ2,      0.0f,
    //        0.0f, 1.0f/2.0f, 1.0f/2.0f,  2.0f,      2.0f,     0.0f,
    //        0.0f, SQ2/4.0f, -SQ2/4.0f,   2.0f*SQ2, -2.0f*SQ2, 1.0f};
    auto multiply_at = [](
        float & o0, float & o1, float & o2, float & o3,
        float i0, float i1, float i2, float i3, float i4, float i5
    ) {
        auto t1p2 = (i1 + i2) * (1.0f / 2.0f);
        auto t1m2 = (i1 - i2) * (SQ2/4.0f);
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
                WinogradTile temp_m;
                for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++) {
                    for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
                        temp_m[xi][nu] =
                            M[(xi*WINOGRAD_ALPHA + nu)*K*P + k*P + b];
                    }
                }
                std::array<std::array<float, WINOGRAD_ALPHA>, WINOGRAD_M> temp;
                std::array<std::array<float, WINOGRAD_M>, WINOGRAD_M> o;

                // Calculates transpose(A).temp_m.A
                for (auto j = 0; j < WINOGRAD_ALPHA; j++){
                    multiply_at(
                        temp[0][j], temp[1][j], temp[2][j], temp[3][j],
                        temp_m[0][j], temp_m[1][j], temp_m[2][j], temp_m[3][j], temp_m[4][j], temp_m[5][j]
                    );
                }

                for (auto i = 0; i < WINOGRAD_M; i++){
                    multiply_at(
                        o[i][0], o[i][1], o[i][2], o[i][3],
                        temp[i][0], temp[i][1], temp[i][2], temp[i][3], temp[i][4], temp[i][5]
                    );
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

void winograd_convolve3::Forward(const int outputs,
                                 const std::vector<float>& input,
                                 const std::vector<float>& U,
                                 std::vector<float>& V,
                                 std::vector<float>& M,
                                 std::vector<float>& output) {

    const auto input_channels = U.size() / (outputs * filter_len);

    winograd_transform_in(input, V, input_channels);
    winograd_sgemm(U, V, M, input_channels, outputs);
    winograd_transform_out(M, output, outputs);
}


std::vector<float> Activation::softmax(const std::vector<float>& input,
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

    for (auto& out : output) {
        out /= denom;
    }

    return output;
}


std::vector<float> Activation::tanh(const std::vector<float>& input) {

	const int input_size = input.size();
	std::vector<float> output;
    output.reserve(input_size);
	
	for (int i = 0; i < input_size; i++) {
		output.push_back(input[i]);
	}

	return output;
}	

template<unsigned int filter_size>
void Convolve<filter_size>::Forward(
              const size_t outputs,
              const std::vector<float>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    // The size of the board is defined at compile time

    const int input_channels = weights.size() / (biases.size() * filter_len);
    const int filter_dim = filter_len * input_channels;
    assert(outputs * num_intersections == output.size());

    std::vector<float> col(filter_dim * width * height);
	
    im2col<filter_size>(input_channels, input, col);
	
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
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M        N            K
                outputs, num_intersections, filter_dim,
                1.0f, &weights[0], filter_dim,
                &col[0], num_intersections,
                0.0f, &output[0], num_intersections);
#else
    auto C_mat = EigenMatrixMap<float>(output.data(),
                                       num_intersections, outputs);
    C_mat.noalias() =
        ConstEigenMatrixMap<float>(col.data(), num_intersections, filter_dim)
        * ConstEigenMatrixMap<float>(weights.data(), filter_dim, outputs);
#endif
	
    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < num_intersections; b++) {
            output[(o * num_intersections) + b] += biases[o];
        }
    }
}

template<>
void Convolve<1>::Forward(
              const size_t outputs,
              const std::vector<float>& input,
              const std::vector<float>& weights,
              const std::vector<float>& biases,
              std::vector<float>& output) {
    Convolve_1::Forward(outputs, input, weights, biases, output);
}

void Convolve_1::Forward(const size_t outputs,
                         const std::vector<float>& input,
                         const std::vector<float>& weights,
                         const std::vector<float>& biases,
                         std::vector<float>& output) {

    const int input_channels = weights.size() / (biases.size());
    const int filter_dim = input_channels;
    assert(outputs * num_intersections == output.size());

	
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
#ifdef USE_BLAS
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                // M        N            K
                outputs, num_intersections, filter_dim,
                1.0f, &weights[0], filter_dim,
                &input[0], num_intersections,
                0.0f, &output[0], num_intersections);
#else
    auto C_mat = EigenMatrixMap<float>(output.data(),
                                       num_intersections, outputs);
    C_mat.noalias() =
        ConstEigenMatrixMap<float>(input.data(), num_intersections, filter_dim)
        * ConstEigenMatrixMap<float>(weights.data(), filter_dim, outputs);
#endif
	
    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < num_intersections; b++) {
            output[(o * num_intersections) + b] += biases[o];
        }
    }
}


static void global_avg_pooling(const size_t channels, const float* input,
                               float* output) {

	constexpr unsigned int width = CONV2D_SIZE;
    constexpr unsigned int height = CONV2D_SIZE;
    constexpr int num_intersections = width * height;

	for (auto c = size_t{0}; c < channels; c++) {
		auto acc = 0.0f;
		for (auto i = size_t{0}; i < num_intersections; i++) {
			acc += input[c * num_intersections + i];
		}
		output[c] = acc / num_intersections;
	}
}

static void apply_se(const size_t channels,
                     const float* input, const float* res, const float* scale,
                     float* output) {

	constexpr unsigned int num_intersections = NUM_INTERSECTIONS;
 
	const auto lambda_ReLU = [](const auto val) {
		return (val > 0.0f) ? val : 0;
	};

	const auto lambda_sigmoid = [](const auto val) {
		return 1.0f / (1.0f + std::exp(-val));
	};

	for (auto c = size_t{0}; c < channels; c++) {
		auto gamma = lambda_sigmoid(scale[c]);
		auto beta  = scale[c];
		for (auto i = size_t{0}; i < num_intersections; i++) {
			output[c * num_intersections + i] = lambda_ReLU(gamma * input[c * num_intersections + i] +
				                                   beta + res[c * num_intersections + i]);
		}
	}
}

void ApplySEUnit(const size_t channels,
                 const size_t se_fc_outputs, const float* input,
                 const float* residual, const float* weights_w1,
                 const float* weights_b1, const float* weights_w2,
                 const float* weights_b2, float* output) {
	/*
	std::vector<float> pool(2 * channels * batch_size);
	std::vector<float> fc_out1(batch_size * se_fc_outputs);

	global_avg_pooling(channels * batch_size, input, pool.data());
	
	innerproduct<>(batch_size, channels, se_fc_outputs,
		           pool.data(), weights_w1, weights_b1,
		           true,  // Relu On
		           fc_out1.data());

	innerproduct<>(batch_size, se_fc_outputs,
		           2 * channels, fc_out1.data(),
		           weights_w2, weights_b2,
		           false,  // Relu Off
		           pool.data());

	*/
}


