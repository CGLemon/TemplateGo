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
#ifndef USE_BLAS
#include <Eigen/Dense>
#endif

#include "NetPipe.h"

#include <vector>
#include <cassert>
#include "config.h"

#ifndef USE_BLAS
// Eigen helpers
template <typename T>
using EigenMatrixMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;
template <typename T>
using ConstEigenMatrixMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>>;

template <typename T>
using EigenVectorMap =
    Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, 1>>;
template <typename T>
using ConstEigenVectorMap =
    Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, 1>>;
#endif




template<unsigned long filter_size>
void im2col(const int channels,
            const std::vector<float>& input,
            std::vector<float>& output);


std::vector<float> winograd_transform_f(const std::vector<float>& f,
                                                   const int outputs, const int channels);


void winograd_transform_in(const std::vector<float>& in,
                               std::vector<float>& V,
                               const int C);

void winograd_sgemm(const std::vector<float>& U,
                    const std::vector<float>& V,
                    std::vector<float>& M,
                    const int C, const int K);

void winograd_transform_out(const std::vector<float>& M,
                            std::vector<float>& Y,
                            const int K);

class winograd_convolve3 {
public:
	static void Forward(const int outputs,
		         const std::vector<float>& input,
		         const std::vector<float>& U,
		         std::vector<float>& V,
		         std::vector<float>& M,
		         std::vector<float>& output);

private:
	 static constexpr unsigned int filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;

};

template<unsigned int filter_size>
class Convolve {
public:
	static void Forward(const size_t output_channels, 
		                const std::vector<float>& input,
		                const std::vector<float>& weights,
		                const std::vector<float>& biases,
		                std::vector<float>& output);
private:
	static constexpr int width = CONV2D_SIZE;
    static constexpr int height = CONV2D_SIZE; 
	static constexpr int num_intersections = width * height;
	static constexpr int filter_len = filter_size * filter_size;
};

class Convolve_1 {
public:
	static void Forward(const size_t outputs, 
                        const std::vector<float>& input,
                        const std::vector<float>& weights,
                        const std::vector<float>& biases,
                        std::vector<float>& output);
private:
	static constexpr int width = CONV2D_SIZE;
    static constexpr int height = CONV2D_SIZE; 
	static constexpr int num_intersections = width * height; 
};

class Activation {
public:
	static std::vector<float> softmax(const std::vector<float>& input,
		                              const float temperature = 1.0f);
	static std::vector<float> tanh(const std::vector<float>& input);

};


template<unsigned int spatial_size>
class Batchnorm {
public:
	static void Forward(const size_t channels,
		                std::vector<float>& data,
		                const float* const means,
		                const float* const stddevs,
		                const float* const eltwise = nullptr);
private:
	static constexpr int num_intersections  = spatial_size;
};

/*
template<unsigned int inputs,
         unsigned int outputs,
         bool ReLU, size_t W>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::array<float, W>& weights,
                                const std::array<float, outputs>& biases);
*/

template<unsigned int inputs,
         unsigned int outputs,
         bool ReLU>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::vector<float>& weights,
                                const std::vector<float>& biases);



template<unsigned int inputs,
         unsigned int outputs,
         bool ReLU, size_t W>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::array<float, W>& weights,
                                const std::array<float, outputs>& biases);



void ApplySEUnit(const size_t channels,
                 const size_t se_fc_outputs, const float* input,
                 const float* residual, const float* weights_w1,
                 const float* weights_b1, const float* weights_w2,
                 const float* weights_b2, float* output);


template<unsigned int spatial_size>
void Batchnorm<spatial_size>::Forward(
               const size_t channels,
               std::vector<float>& data,
               const float* const means,
               const float* const stddevs,
               const float* const eltwise) {
    const auto lambda_ReLU = [=](const auto val) { 
		return (val > 0.0f) ? val : 0.0f;
	};
    for (auto c = size_t{0}; c < channels; ++c) {
        const auto mean = means[c];
        const auto scale_stddev = stddevs[c];
        const auto arr = &data[c * num_intersections];

        if (eltwise == nullptr) {
            // Classical BN
            for (auto b = size_t{0}; b < num_intersections; b++) {
                arr[b] = lambda_ReLU(scale_stddev * (arr[b] - mean));
            }
        } else {
            // BN + residual add
            const auto res = &eltwise[c * num_intersections];
            for (auto b = size_t{0}; b < num_intersections; b++) {
                arr[b] = lambda_ReLU((scale_stddev * (arr[b] - mean)) + res[b]);
            }
        }
    }
}


template<unsigned long filter_size>
void im2col(const int channels,
            const std::vector<float>& input,
            std::vector<float>& output) {
    constexpr unsigned int height = CONV2D_SIZE;
    constexpr unsigned int width = CONV2D_SIZE;

    constexpr int pad = (filter_size / 2);
    constexpr unsigned int output_h = height + 2 * pad - filter_size  + 1;
    constexpr unsigned int output_w = width + 2 * pad - filter_size + 1;

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

/*
template<unsigned int inputs,
         unsigned int outputs,
         bool ReLU,
         size_t W>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::array<float, W>& weights,
                                const std::array<float, outputs>& biases) {
    std::vector<float> output(outputs);

#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);
#else
    EigenVectorMap<float> y(output.data(), outputs);
    y.noalias() =
        ConstEigenMatrixMap<float>(weights.data(),
                                   inputs,
                                   outputs).transpose()
        * ConstEigenVectorMap<float>(input.data(), inputs);
#endif
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (unsigned int o = 0; o < outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }

    return output;
}
*/



template<unsigned int inputs,
         unsigned int outputs,
         bool ReLU>
std::vector<float> innerproduct(const std::vector<float>& input,
                                const std::vector<float>& weights,
                                const std::vector<float>& biases) {
    std::vector<float> output(outputs);

#ifdef USE_BLAS
    cblas_sgemv(CblasRowMajor, CblasNoTrans,
                // M     K
                outputs, inputs,
                1.0f, &weights[0], inputs,
                &input[0], 1,
                0.0f, &output[0], 1);
#else
    EigenVectorMap<float> y(output.data(), outputs);
    y.noalias() =
        ConstEigenMatrixMap<float>(weights.data(),
                                   inputs,
                                   outputs).transpose()
        * ConstEigenVectorMap<float>(input.data(), inputs);
#endif
    const auto lambda_ReLU = [](const auto val) { return (val > 0.0f) ?
                                                          val : 0.0f; };
    for (unsigned int o = 0; o < outputs; o++) {
        auto val = biases[o] + output[o];
        if (ReLU) {
            val = lambda_ReLU(val);
        }
        output[o] = val;
    }

    return output;
}
#endif
