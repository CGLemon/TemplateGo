#ifndef CPULAYERS_H_INCLUDE
#define CPULAYERS_H_INCLUDE

#include "NetPipe.h"

#include <vector>
#include <cassert>
#include "config.h"
#include "blas/CPUblas.h"


std::vector<float> winograd_transform_f(const std::vector<float>& f,
                                                   const int outputs, const int channels);


class winograd_convolve3 {
public:
	static void Forward(const int outputs,
		         const std::vector<float>& input,
		         const std::vector<float>& U,
		         std::vector<float>& V,
		         std::vector<float>& M,
		         std::vector<float>& output);

private:
	static void transform_in(const std::vector<float>& in,
		                     std::vector<float>& V,
		                     const int C);

	static void sgemm(const std::vector<float>& U,
		              const std::vector<float>& V,
		              std::vector<float>& M,
		              const int C, const int K);

	static void transform_out(const std::vector<float>& M,
		                      std::vector<float>& Y,
		                      const int K);

	static constexpr unsigned int WTILES = WINOGRAD_WTILES;
	static constexpr unsigned int W = CONV2D_SIZE;
	static constexpr unsigned int H = CONV2D_SIZE;
	static constexpr unsigned int filter_len = WINOGRAD_ALPHA * WINOGRAD_ALPHA;

};


class Convolve {
public:
	static void Forward(const size_t filter_size,
						const size_t output_channels, 
		                const std::vector<float>& input,
		                const std::vector<float>& weights,
		                const std::vector<float>& biases,
		                std::vector<float>& output);
private:

	static void im2col(const size_t filter_size,
                       const int channels,
                       const std::vector<float>& input,
                       std::vector<float>& output);

	static constexpr int width = CONV2D_SIZE;
    static constexpr int height = CONV2D_SIZE; 
	static constexpr int spatial_size = width * height;
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
	static constexpr int spatial_size = width * height; 
};

class Activation {
public:
	static std::vector<float> softmax(const std::vector<float>& input,
		                              const float temperature = 1.0f);
	static std::vector<float> tanh(const std::vector<float>& input);

};

class Batchnorm {
public:
	static void Forward(const size_t channels,
		                std::vector<float>& data,
		                const float* const means,
		                const float* const stddevs,
		                const float* const eltwise = nullptr);
private:
	static constexpr int spatial_size  = NUM_INTERSECTIONS;
};



class FullyConnect {
public:
	static void Forward(const int inputs,
                        const int outputs,
                        const std::vector<float>& input,
                        const std::vector<float>& weights,
                        const std::vector<float>& biases,
                        std::vector<float>& output,
                        bool ReLU);
};




void ApplySEUnit(const size_t channels,
                 const size_t se_fc_outputs, const float* input,
                 const float* residual, const float* weights_w1,
                 const float* weights_b1, const float* weights_w2,
                 const float* weights_b2, float* output);


#endif
