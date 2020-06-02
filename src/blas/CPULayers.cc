#include "blas/CPULayers.h"
#include "NetPipe.h"

#include <algorithm>
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


void winograd_convolve3::transform_in(const std::vector<float>& in,
                                      std::vector<float>& V,
                                      const int C) {

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

void winograd_convolve3::sgemm(const std::vector<float>& U,
                               const std::vector<float>& V,
                               std::vector<float>& M,
                               const int C, const int K) {
    constexpr auto P = WINOGRAD_P;

    for (int b = 0; b < WINOGRAD_TILE; b++) {
        const int offset_u = b * K * C;
        const int offset_v = b * C * P;
        const int offset_m = b * K * P;
		
		blas::winograd_gemm(
						offset_u, offset_v, offset_m,
						K, P, C,
		                1.0f,
		                U, K,
		                V, P,
		                0.0f,
		                M, P);
		


    }
}

void winograd_convolve3::transform_out(const std::vector<float>& M,
                                       std::vector<float>& Y,
                                       const int K) {
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

    transform_in(input, V, input_channels);
    sgemm(U, V, M, input_channels, outputs);
    transform_out(M, output, outputs);
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



void Convolve::im2col(const size_t filter_size,
                      const int channels,
                      const std::vector<float>& input,
                      std::vector<float>& output) {

    int pad = (filter_size / 2);
    unsigned int output_h = height + 2 * pad - filter_size  + 1;
    unsigned int output_w = width + 2 * pad - filter_size + 1;

    const float* data_im = input.data();
    float* data_col = output.data();

    for (int channel = channels; channel--; data_im += NUM_INTERSECTIONS) {
        for (unsigned int kernel_row = 0; kernel_row < filter_size; kernel_row++) {
            for (unsigned int kernel_col = 0; kernel_col < filter_size; kernel_col++) {
                int input_row = -pad + kernel_row;
                for (int output_rows = output_h; output_rows; output_rows--) {
                    if (unsigned(input_row) < height) {
                        int input_col = -pad + kernel_col;
                        for (int output_col = output_w; output_col; output_col--) {
                            if (unsigned(input_col) < width) {
                                *(data_col++) =
                                    data_im[input_row * width + input_col];
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


void Convolve::Forward(const size_t filter_size,
                       const size_t outputs,
                       const std::vector<float>& input,
                       const std::vector<float>& weights,
                       const std::vector<float>& biases,
                       std::vector<float>& output) {
	int filter_len = filter_size * filter_size;
    const int input_channels = weights.size() / (biases.size() * filter_len);
    const int filter_dim = filter_len * input_channels;
    assert(outputs * spatial_size == output.size());

    std::vector<float> col(filter_dim * width * height);
	
    im2col(filter_size, input_channels, input, col);

	blas::fixed_gemm(outputs, spatial_size, filter_dim,
                     1.0f, weights, filter_dim,
                           col, spatial_size,
                     0.0f, output, spatial_size);
	
    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < spatial_size; b++) {
            output[(o * spatial_size) + b] += biases[o];
        }
    }
}


void Convolve_1::Forward(const size_t outputs,
                         const std::vector<float>& input,
                         const std::vector<float>& weights,
                         const std::vector<float>& biases,
                         std::vector<float>& output) {

    const int input_channels = weights.size() / (biases.size());
    const int filter_dim = input_channels;
    assert(outputs * spatial_size == output.size());

	blas::fixed_gemm(outputs, spatial_size, filter_dim,
                     1.0f, weights, filter_dim,
                           input, spatial_size,
                     0.0f, output, spatial_size);

    for (unsigned int o = 0; o < outputs; o++) {
        for (unsigned int b = 0; b < spatial_size; b++) {
            output[(o * spatial_size) + b] += biases[o];
        }
    }
}

void Batchnorm::Forward(
               const size_t channels,
               std::vector<float>& data,
               const float* const means,
               const float* const stddevs,
               const float* const eltwise) {
	assert(spatial_size*channels == data.size());
    const auto lambda_ReLU = [=](const auto val) { 
		return (val > 0.0f) ? val : 0.0f;
	};
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddev = stddevs[c];
        const auto arr = &data[c * spatial_size];

        if (eltwise == nullptr) {
            // Classical BN
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU(scale_stddev * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            const auto res = &eltwise[c * spatial_size];
            for (auto b = size_t{0}; b < spatial_size; b++) {
                arr[b] = lambda_ReLU((scale_stddev * (arr[b] - mean)) + res[b]);
            }
        }
    }
}

void FullyConnect::Forward(const int inputs,
                           const int outputs,
                           const std::vector<float>& input,
                           const std::vector<float>& weights,
                           const std::vector<float>& biases,
                           std::vector<float>& output,
                           bool ReLU) {


	blas::dense(inputs, outputs, input, weights, output);
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (unsigned int o = 0; o < outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
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
