#ifndef CUDABACKEND_H_INCLUDE
#define CUDABACKEND_H_INCLUDE

#include "NetPipe.h"

class CUDAbackend : public ForwardPipe {
public:
	virtual void initialize(const int channels, int residual_blocks,
                              std::shared_ptr<ForwardPipeWeights> weights);
    
    virtual void forward(const std::vector<float>& input,
                         std::vector<float>& output_pol,
                         std::vector<float>& output_val);
    virtual void push_weights(unsigned int filter_size,
                              unsigned int channels,
                              unsigned int outputs,
                              std::shared_ptr<const ForwardPipeWeights> weights);



private:
	int m_input_channels;
	int m_residual_blocks;

};

#endif
