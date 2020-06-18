#include "CUDAbackend.h"
#ifdef USE_CUDA
#include "cuda/CUDALayers.h"
#include "cuda/CUDACommon.h"
#include "cuda/CUDAKernels.h"
#include "Utils.h"
#include <cassert>

using namespace Utils;



void OutputSpec(const cudaDeviceProp sDevProp) {
  auto_printf("Device name: %s\n", sDevProp.name);
  auto_printf("Device memory(MiB): %zu\n", (sDevProp.totalGlobalMem/(1024*1024)));
  auto_printf("Memory per-block(KiB): %zu\n", (sDevProp.sharedMemPerBlock/1024));
  auto_printf("Register per-block(KiB): %zu\n", (sDevProp.regsPerBlock/1024));
  auto_printf("Warp size: %zu\n", sDevProp.warpSize);
  auto_printf("Memory pitch(MiB): %zu\n", (sDevProp.memPitch/(1024*1024)));
  auto_printf("Constant Memory(KiB): %zu\n", (sDevProp.totalConstMem/1024));
  auto_printf("Max thread per-block: %zu\n", sDevProp.maxThreadsPerBlock);
  auto_printf("Max thread dim: (%zu, %zu, %zu)\n", sDevProp.maxThreadsDim[0], sDevProp.maxThreadsDim[1], sDevProp.maxThreadsDim[2]);
  //auto_printf("Max grid size: (%zu, %zu, %zu)\n", sDevProp.maxGridSize[0], sDevProp.maxGridSize[1], sDevProp.maxGridSize[2]);
  auto_printf("Ver: %zu.%zu\n", sDevProp.major, sDevProp.minor);
  auto_printf("Clock: %zu(kHz)\n", (sDevProp.clockRate/1000));
  auto_printf("textureAlignment: %zu\n", sDevProp.textureAlignment);
}


CUDAbackend::CUDAbackend() {
  auto_printf("Using CUDA network.\n");
  int iDeviceCount = 0;
  cudaGetDeviceCount(&iDeviceCount);
  auto_printf("Number of CUDA devices: %zu\n", iDeviceCount);

  if(iDeviceCount == 0) {
    auto_printf("No CUDA device\n");
    exit(0);
  }

  for(int i = 0; i < iDeviceCount; ++ i) {
    auto_printf("\n=== Device %i ===\n", i);
    cudaDeviceProp sDeviceProp;
    cudaGetDeviceProperties(&sDeviceProp, i);
    OutputSpec(sDeviceProp);
  }
  auto_printf("\n");
  cudaSetDevice(0);

  is_loaded = false;
}

CUDAbackend::~CUDAbackend() {
  if(is_loaded) {
    ReportCUDAErrors(cudaFree(cuda_input));
    ReportCUDAErrors(cudaFree(cuda_output_pol));
    ReportCUDAErrors(cudaFree(cuda_output_val));
    ReportCUDAErrors(cudaFree(cuda_conv_out));
    ReportCUDAErrors(cudaFree(cuda_conv_in));
    ReportCUDAErrors(cudaFree(cuda_res));
  }
}


void CUDAbackend::initialize(const int channels, int residual_blocks,
                             std::shared_ptr<ForwardPipeWeights> weights) {
  m_input_channels = channels;
  m_residual_blocks = residual_blocks;
}

void CUDAbackend::forward(const std::vector<float> &input,
                          std::vector<float> &output_pol,
                          std::vector<float> &output_val) {

  int batch = 1;
  size_t output_channels = m_input_channels;

  size_t input_s = INPUT_CHANNELS * NUM_INTERSECTIONS * sizeof(float);
  size_t output_pol_s = OUTPUTS_POLICY * NUM_INTERSECTIONS * sizeof(float);
  size_t output_val_s = OUTPUTS_VALUE * NUM_INTERSECTIONS * sizeof(float);

  size_t swap_s = output_channels * NUM_INTERSECTIONS;

  ReportCUDAErrors(cudaMemcpy(cuda_input, input.data(), input_s, cudaMemcpyHostToDevice));

  conv_layer[0].Forward(batch, cuda_input, cuda_conv_out);
  batch_layer[0].Forward(batch, cuda_conv_out);
  
  for (int i = 1; i < (2*m_residual_blocks+1); i+=2) {
    cuda_swap(cuda_conv_out, cuda_conv_in, swap_s);
    conv_layer[i].Forward(batch, cuda_conv_in, cuda_conv_out);
    batch_layer[i].Forward(batch, cuda_conv_out);

    cuda_swap(cuda_conv_in, cuda_res, swap_s);
    cuda_swap(cuda_conv_out, cuda_conv_in, swap_s);

    conv_layer[i+1].Forward(batch, cuda_conv_in, cuda_conv_out);
    batch_layer[i+1].Forward(batch, cuda_conv_out, cuda_res);
  }
  conv_head[0].Forward(batch, cuda_conv_out, cuda_output_pol);
  conv_head[1].Forward(batch, cuda_conv_out, cuda_output_val); 


  ReportCUDAErrors(cudaMemcpy(output_pol.data(), cuda_output_pol, output_pol_s, cudaMemcpyDeviceToHost));
  ReportCUDAErrors(cudaMemcpy(output_val.data(), cuda_output_val, output_val_s, cudaMemcpyDeviceToHost));
}

void CUDAbackend::push_weights(
    unsigned int filter_size, unsigned int channels, unsigned int outputs,
    std::shared_ptr<const ForwardPipeWeights> weights) {


  conv_layer.emplace_back(1, 3, INPUT_CHANNELS, m_input_channels);
  batch_layer.emplace_back(1, m_input_channels);

  for (int i = 0; i < m_residual_blocks; ++i) {
    conv_layer.emplace_back(1, 3, m_input_channels, m_input_channels);
    batch_layer.emplace_back(1, m_input_channels);
    conv_layer.emplace_back(1, 3, m_input_channels, m_input_channels);
    batch_layer.emplace_back(1, m_input_channels);
  }

  assert(2*m_residual_blocks+1 == weights->m_conv_weights.size());

  conv_head.emplace_back(1, 1, m_input_channels, OUTPUTS_POLICY);
  conv_head.emplace_back(1, 1, m_input_channels, OUTPUTS_VALUE);

  for (int i = 0; i < (2*m_residual_blocks+1); ++i) {
    conv_layer[i].LoadingWeight(weights->m_conv_weights[i]);
    batch_layer[i].LoadingWeight(weights->m_batchnorm_means[i], weights->m_batchnorm_stddevs[i]);
  }

  conv_head[0].LoadingWeight(weights->m_conv_pol_w);
  conv_head[1].LoadingWeight(weights->m_conv_val_w);

  size_t output_channels = m_input_channels;

  size_t input_s = INPUT_CHANNELS * NUM_INTERSECTIONS * sizeof(float);
  size_t output_pol_s = OUTPUTS_POLICY * NUM_INTERSECTIONS * sizeof(float);
  size_t output_val_s = OUTPUTS_VALUE * NUM_INTERSECTIONS * sizeof(float);

  size_t conv_s = output_channels * NUM_INTERSECTIONS * sizeof(float);

  ReportCUDAErrors(cudaMalloc(&cuda_input, input_s));
  ReportCUDAErrors(cudaMalloc(&cuda_output_pol, output_pol_s));
  ReportCUDAErrors(cudaMalloc(&cuda_output_val, output_val_s));
  ReportCUDAErrors(cudaMalloc(&cuda_conv_out, conv_s));
  ReportCUDAErrors(cudaMalloc(&cuda_conv_in, conv_s));
  ReportCUDAErrors(cudaMalloc(&cuda_res, conv_s));

}

#endif
