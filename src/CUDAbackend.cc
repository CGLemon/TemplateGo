#include "CUDAbackend.h"

void CUDAbackend::initialize(const int channels, int residual_blocks,
                        std::shared_ptr<ForwardPipeWeights> weights) {
	
	int m_input_channels = channels;
	int m_residual_blocks = residual_blocks;

}


void CUDAbackend::forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val) {


}

void CUDAbackend::push_weights(unsigned int filter_size,
                              unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights){


}
