#include "LZ/LZCUDAbackend.h"
#include "Utils.h"
#include "cfg.h"

#ifdef USE_CUDA

using namespace Utils;

namespace LZ {

CUDAbackend::CUDAbackend() {

  is_applied = false;
  auto_printf("Using CUDA network.\n");
}

CUDAbackend::~CUDAbackend() {
  if(is_applied) {
    ReportCUDAErrors(cudaFree(cuda_input));
    ReportCUDAErrors(cudaFree(cuda_output_pol));
    ReportCUDAErrors(cudaFree(cuda_output_val));

    ReportCUDAErrors(cudaFree(cuda_conv_temp[0]));
    ReportCUDAErrors(cudaFree(cuda_conv_temp[1]));
    ReportCUDAErrors(cudaFree(cuda_conv_temp[2]));

    ReportCUDAErrors(cudaFree(cuda_pol_layer[0]));
    ReportCUDAErrors(cudaFree(cuda_val_layer[0]));
    ReportCUDAErrors(cudaFree(cuda_val_layer[1]));
#ifdef USE_CUDNN
    ReportCUDAErrors(cudaFree(cuda_scratch));
#endif
  }
}

void CUDAbackend::initialize(std::shared_ptr<NNWeights> weights) {
  m_residual_channels = weights->get_num_channels(0);
  m_residual_blocks = weights->get_num_residuals();
  m_weights = weights;
  LZModel::transform(false, m_weights);

  pushing_weights(weights);

  handel.apply();
  void * cuda_scratch = nullptr;
  size_t m_scratch_size = 0;

  cuda_gpu_info();
}

// TODO: 支持多 batch 
void CUDAbackend::forward(const std::vector<float> &input,
                          std::vector<float> &output_pol,
                          std::vector<float> &output_val) {

  std::lock_guard<std::mutex> lock(m_mtx);
  batch_forward(1, input, output_pol, output_val);
}

void CUDAbackend::batch_forward(const size_t batch,
                                const std::vector<float> &input,
                                std::vector<float> &output_pol,
                                std::vector<float> &output_val) {
  assert(batch == 1);
  const size_t input_s = batch * LZ::INPUT_CHANNELS * NUM_INTERSECTIONS * sizeof(float);
  const size_t output_pol_s = batch * LZ::POTENTIAL_MOVES * sizeof(float);
  const size_t output_val_s = batch * LZ::VALUE_LABELS * sizeof(float);
  const size_t copy_s = batch * m_residual_channels * NUM_INTERSECTIONS;

  ReportCUDAErrors(cudaMemcpy(cuda_input, input.data(),
                              input_s, cudaMemcpyHostToDevice));
  
  conv_layer[0].Forward(batch, cuda_input, cuda_conv_temp[0],
                        cuda_scratch, m_scratch_size, &handel);
  bnorm_layer[0].Forward(batch, cuda_conv_temp[0]);

  // Residual tower
  for (int i = 1; i < (2*m_residual_blocks+1); i+=2) {
    conv_layer[i].Forward(batch, cuda_conv_temp[0], cuda_conv_temp[1],
                          cuda_scratch, m_scratch_size, &handel);
    bnorm_layer[i].Forward(batch, cuda_conv_temp[1]);

    conv_layer[i+1].Forward(batch, cuda_conv_temp[1], cuda_conv_temp[2],
                            cuda_scratch, m_scratch_size, &handel);
    bnorm_layer[i+1].Forward(batch, cuda_conv_temp[2], cuda_conv_temp[0]);
    cuda_copy(cuda_conv_temp[0], cuda_conv_temp[2], copy_s);
  }

  // policy head
  poliy_conv.Forward(batch, cuda_conv_temp[0], cuda_pol_layer[0],
                     cuda_scratch, m_scratch_size, &handel);
  poliy_bnorm.Forward(batch, cuda_pol_layer[0]);
  poliy_fc.Forward(batch, cuda_pol_layer[0], cuda_output_pol, &handel);

  // value head
  value_conv.Forward(batch, cuda_conv_temp[0], cuda_val_layer[0],
                     cuda_scratch, m_scratch_size, &handel);
  value_bnorm.Forward(batch, cuda_val_layer[0]);
  value_fc1.Forward(batch, cuda_val_layer[0], cuda_val_layer[1], &handel);
  value_fc2.Forward(batch, cuda_val_layer[1], cuda_output_val, &handel);
  
  ReportCUDAErrors(cudaMemcpy(output_pol.data(), cuda_output_pol,
                              output_pol_s, cudaMemcpyDeviceToHost));
  ReportCUDAErrors(cudaMemcpy(output_val.data(), cuda_output_val,
                              output_val_s, cudaMemcpyDeviceToHost));
}



void CUDAbackend::pushing_weights(std::shared_ptr<NNWeights> weights) {
  const size_t residual_channels = weights->get_num_channels(0);
  const size_t residual_blocks = weights->get_num_residuals();
  const size_t max_batchsize = cfg_batchsize;

  const size_t filter_3 = RESIDUAL_FILTER;
  const size_t filter_1 = HEAD_FILTER;

// TODO : Load form descriptor

  conv_layer.emplace_back(max_batchsize, filter_3,
                          LZ::INPUT_CHANNELS, residual_channels);
  bnorm_layer.emplace_back(max_batchsize, residual_channels);

  // Residual tower
  for (auto i = size_t{0}; i < residual_blocks; ++i) {
    conv_layer.emplace_back(max_batchsize, filter_3,
                            residual_channels, residual_channels);
    bnorm_layer.emplace_back(max_batchsize, residual_channels);


    conv_layer.emplace_back(max_batchsize, filter_3,
                            residual_channels, residual_channels);
    bnorm_layer.emplace_back(max_batchsize, residual_channels);
  }

  // policy head
  poliy_conv = CudaConvolve(max_batchsize, filter_1,
                            residual_channels, LZ::OUTPUTS_POLICY);
  poliy_bnorm = CudaBatchnorm(max_batchsize, LZ::OUTPUTS_POLICY);
  poliy_fc = CudaFullyConnect(max_batchsize, 
                              LZ::OUTPUTS_POLICY * NUM_INTERSECTIONS, 
                              LZ::POTENTIAL_MOVES, false); 
  
  // value head
  value_conv = CudaConvolve(max_batchsize, filter_1,
                            residual_channels, LZ::OUTPUTS_VALUE);
  value_bnorm = CudaBatchnorm(max_batchsize, LZ::OUTPUTS_VALUE);
  value_fc1 = CudaFullyConnect(max_batchsize, 
                               LZ::OUTPUTS_VALUE * NUM_INTERSECTIONS,
                               LZ::VALUE_LAYER, true);
  value_fc2 = CudaFullyConnect(max_batchsize, 
                               LZ::VALUE_LAYER,
                               LZ::VALUE_LABELS, false);


  conv_layer[0].LoadingWeight(m_weights->m_ip_conv.m_conv.m_weights, m_scratch_size);
  bnorm_layer[0].LoadingWeight(m_weights->m_ip_conv.m_batchnorm.m_means,
                               m_weights->m_ip_conv.m_batchnorm.m_stddevs);
  // Residual tower
  for (auto i = size_t{0}; i < residual_blocks; ++i) {
    conv_layer[2*i+1].LoadingWeight(m_weights->m_res_blocks[i].m_conv_blocks[0].m_conv.m_weights, m_scratch_size);
    bnorm_layer[2*i+1].LoadingWeight(m_weights->m_res_blocks[i].m_conv_blocks[0].m_batchnorm.m_means,
                                     m_weights->m_res_blocks[i].m_conv_blocks[0].m_batchnorm.m_stddevs);

    conv_layer[2*i+2].LoadingWeight(m_weights->m_res_blocks[i].m_conv_blocks[1].m_conv.m_weights, m_scratch_size);
    bnorm_layer[2*i+2].LoadingWeight(m_weights->m_res_blocks[i].m_conv_blocks[1].m_batchnorm.m_means,
                                     m_weights->m_res_blocks[i].m_conv_blocks[1].m_batchnorm.m_stddevs);
  }
   // policy head
  poliy_conv.LoadingWeight(m_weights->m_conv_pol.m_conv.m_weights, m_scratch_size);
  poliy_bnorm.LoadingWeight(m_weights->m_conv_pol.m_batchnorm.m_means,
                            m_weights->m_conv_pol.m_batchnorm.m_stddevs);
  poliy_fc.LoadingWeight(m_weights->m_fc_pol.m_weights,
                         m_weights->m_fc_pol.m_biases);

  // value head
  value_conv.LoadingWeight(m_weights->m_conv_val.m_conv.m_weights, m_scratch_size);
  value_bnorm.LoadingWeight(m_weights->m_conv_val.m_batchnorm.m_means,
                            m_weights->m_conv_val.m_batchnorm.m_stddevs);
  value_fc1.LoadingWeight(m_weights->m_fc1_val.m_weights,
                          m_weights->m_fc1_val.m_biases);
  value_fc2.LoadingWeight(m_weights->m_fc2_val.m_weights,
                          m_weights->m_fc2_val.m_biases);

  assert(max_batchsize == 1);
  const size_t input_s 
                 = max_batchsize * LZ::INPUT_CHANNELS * NUM_INTERSECTIONS * sizeof(float);
  const size_t conv_s 
                 = max_batchsize * residual_channels * NUM_INTERSECTIONS * sizeof(float);
  const size_t output_pol_s
                 = max_batchsize * LZ::POTENTIAL_MOVES * sizeof(float);
  const size_t output_val_s 
                 = max_batchsize * LZ::VALUE_LABELS * sizeof(float);
  const size_t pol_layer_s 
                 = max_batchsize * LZ::OUTPUTS_POLICY * NUM_INTERSECTIONS * sizeof(float);

  const size_t val_layer1_s 
                 = max_batchsize * LZ::OUTPUTS_VALUE * NUM_INTERSECTIONS * sizeof(float);
  const size_t val_layer2_s 
                 = max_batchsize * LZ::VALUE_LAYER * sizeof(float);

  ReportCUDAErrors(cudaMalloc(&cuda_input, input_s));
  ReportCUDAErrors(cudaMalloc(&cuda_output_pol, output_pol_s));
  ReportCUDAErrors(cudaMalloc(&cuda_output_val, output_val_s));
  ReportCUDAErrors(cudaMalloc(&cuda_conv_temp[0], conv_s));
  ReportCUDAErrors(cudaMalloc(&cuda_conv_temp[1], conv_s));
  ReportCUDAErrors(cudaMalloc(&cuda_conv_temp[2], conv_s));

  ReportCUDAErrors(cudaMalloc(&cuda_pol_layer[0], pol_layer_s));
  ReportCUDAErrors(cudaMalloc(&cuda_val_layer[0], val_layer1_s));
  ReportCUDAErrors(cudaMalloc(&cuda_val_layer[1], val_layer2_s));
  
#ifdef USE_CUDNN
  ReportCUDAErrors(cudaMalloc(&cuda_scratch, m_scratch_size));
#endif

  is_applied = true;
}
}


#endif
