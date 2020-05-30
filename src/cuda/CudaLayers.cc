/*
  This file is part of Leela Chess Zero.
  Copyright (C) 2018-2019 The LCZero Authors

  Leela Chess is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Leela Chess is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with Leela Chess.  If not, see <http://www.gnu.org/licenses/>.

  Additional permission under GNU GPL version 3 section 7

  If you modify this Program, or any covered work, by linking or
  combining it with NVIDIA Corporation's libraries from the NVIDIA CUDA
  Toolkit and the NVIDIA CUDA Deep Neural Network library (or a
  modified version of those libraries), containing parts covered by the
  terms of the respective license agreement, the licensors of this
  Program grant you additional permission to convey the resulting work.
*/
#include "cuda/CudaLayers.h"

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip, bool nhwc)
    : input_(ip), C(c), H(h), W(w), nhwc_(nhwc) {}

template <typename DataType>
BaseLayer<DataType>::BaseLayer(int c, int h, int w, BaseLayer* ip)
    : input_(ip), C(c), H(h), W(w), nhwc_(ip->nhwc_) {}


static constexpr bool kUseFusedSELayer = true;

template <typename DataType>
void ConvLayer<DataType>::init() {
  // Allocate memory for weights (filter tensor) and biases.
  const size_t weight_size =
      sizeof(DataType) * c_input_ * C * filter_size_ * filter_size_;
  ReportCUDAErrors(cudaMalloc(&weights, weight_size));

  const size_t blas_size = sizeof(DataType) * C;
  ReportCUDAErrors(cudaMalloc(&biases, blas_size));

  const bool fp16 = std::is_same<half, DataType>::value;
  const cudnnDataType_t dataType =
      std::is_same<half, DataType>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

  const cudnnTensorFormat_t layout =
      nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  // Create cudnn objects for various tensors, algorithms, etc.
  cudnnCreateFilterDescriptor(&filter_desc_);
  cudnnCreateConvolutionDescriptor(&conv_desc_);
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
  cudnnCreateTensorDescriptor(&in_tensor_desc_);
  cudnnCreateTensorDescriptor(&bias_desc_);
  cudnnCreateActivationDescriptor(&activation_);

  cudnnSetFilter4dDescriptor(filter_desc_, dataType, layout, GetC(), c_input_,
                             filter_size_, filter_size_);

  ReportCUDNNErrors(
      cudnnSetTensor4dDescriptor(bias_desc_, layout, dataType, 1, C, 1, 1));

  const int padding = filter_size_ / 2;
  const bool crossCorr = 1;

  ReportCUDNNErrors(cudnnSetConvolution2dDescriptor(
      conv_desc_, padding, padding, 1, 1, 1, 1,
      crossCorr ? CUDNN_CROSS_CORRELATION : CUDNN_CONVOLUTION, dataType));

  if (fp16 && nhwc_)
    ReportCUDNNErrors(
        cudnnSetConvolutionMathType(conv_desc_, CUDNN_TENSOR_OP_MATH));

  // TODO: dynamic selection of algorithm!
  if ((C > 32) && (!nhwc_) && (filter_size_ > 1)) {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_WINOGRAD_NONFUSED;
  } else {
    conv_algo_ = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
  }

  if (use_relu_) {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_RELU,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
#if CUDNN_MAJOR != 7 || CUDNN_MINOR != 0
  else {
    cudnnSetActivationDescriptor(activation_, CUDNN_ACTIVATION_IDENTITY,
                                 CUDNN_NOT_PROPAGATE_NAN, 0.0);
  }
#endif
}

template <typename DataType>
ConvLayer<DataType>::ConvLayer(BaseLayer<DataType>* ip, int C, int H, int W,
                               int filter, int Cin, bool relu, bool bias)
    : BaseLayer<DataType>(C, H, W, ip),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
      use_bias_(bias) {
  init();
}

template <typename DataType>
ConvLayer<DataType>::ConvLayer(bool nhwc, int C, int H, int W, int filter,
                               int Cin, bool relu, bool bias)
    : BaseLayer<DataType>(C, H, W, nullptr, nhwc),
      c_input_(Cin),
      filter_size_(filter),
      use_relu_(relu),
      use_bias_(bias) {
  init();
}
/*
template <>
void ConvLayer<half>::LoadWeights(float* pfilter, float* pBias, void* scratch) {
  const size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  const size_t blas_size = sizeof(float) * C;
  // Also need to convert from fp32 NCHW to fp16 NHWC
  // first copy from CPU memory to scratch space in GPU memory
  // and then do the type / layout conversion using a kernel.
  assert(scratch);
  ReportCUDAErrors(
      cudaMemcpy(scratch, pfilter, weight_size, cudaMemcpyHostToDevice));

  if (nhwc_) {
    fp32NCHWtofp16NHWC((half*)weights, (float*)scratch, C, c_input_, C,
                       c_input_, filter_size_, filter_size_);
  } else {
    copyTypeConverted((half*)weights, (float*)scratch,
                      C * c_input_ * filter_size_ * filter_size_);
  }

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(scratch, pBias, blas_size, cudaMemcpyHostToDevice));

    copyTypeConverted((half*)biases, (float*)scratch, C);
  }
}
*/

template <>
void ConvLayer<float>::LoadWeights(float* pfilter, float* pBias,
                                   void* /*scratch*/) {
  const size_t weight_size =
      sizeof(float) * c_input_ * C * filter_size_ * filter_size_;
  const size_t blas_size = sizeof(float) * C;
  ReportCUDAErrors(
      cudaMemcpy(weights, pfilter, weight_size, cudaMemcpyHostToDevice));

  if (pBias) {
    ReportCUDAErrors(
        cudaMemcpy(biases, pBias, blas_size, cudaMemcpyHostToDevice));
  } else {
    ReportCUDAErrors(cudaMemset(biases, 0, blas_size));
  }
}

template <typename DataType>
void ConvLayer<DataType>::Eval(int N, DataType* output, const DataType* input,
                               const DataType* input2, void* scratch,
                               size_t scratch_size, cudnnHandle_t cudnn,
                               cublasHandle_t /*cublas*/) {
  const cudnnDataType_t dataType =
      std::is_same<half, DataType>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

  const cudnnTensorFormat_t layout =
      nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(out_tensor_desc_, layout,
                                               dataType, N, C, H, W));

  ReportCUDNNErrors(cudnnSetTensor4dDescriptor(in_tensor_desc_, layout,
                                               dataType, N, c_input_, H, W));

  float alpha = 1.0f, beta = 0.0f;

  if (!(use_relu_ || use_bias_ || input2)) {
    ReportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &beta, out_tensor_desc_,
        output));
  }
#if CUDNN_MAJOR != 7 || CUDNN_MINOR != 0
  else if (input2) {
    // fused bias + sum + relu!
    ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size, &alpha, out_tensor_desc_,
        input2, bias_desc_, biases, activation_, out_tensor_desc_, output));
  } else {
    // For some reason cudnn doesn't support just Convolution + Bias with nchw
    // (winograd algorithm) it works fine when RELU is also needed which is
    // somewhat strange.
    if ((!nhwc_) && (!use_relu_)) {
      ReportCUDNNErrors(cudnnConvolutionForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output));
      // add bias
      addBias_NCHW(output, output, biases, N, C, H, W);
    } else {
      ReportCUDNNErrors(cudnnConvolutionBiasActivationForward(
          cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
          conv_desc_, conv_algo_, scratch, scratch_size, &beta,
          out_tensor_desc_, output, bias_desc_, biases, activation_,
          out_tensor_desc_, output));
    }
  }
#else
  else {
    ReportCUDNNErrors(cudnnConvolutionForward(
        cudnn, &alpha, in_tensor_desc_, input, filter_desc_, weights,
        conv_desc_, conv_algo_, scratch, scratch_size,
        (input2 == output) ? &alpha : &beta, out_tensor_desc_, output));
    if (input2 && input2 != output) {
      ReportCUDNNErrors(cudnnAddTensor(cudnn, &alpha, out_tensor_desc_, input2,
                                       &alpha, out_tensor_desc_, output));
    }
    if (use_bias_) {
      ReportCUDNNErrors(cudnnAddTensor(cudnn, &alpha, bias_desc_, biases,
                                       &alpha, out_tensor_desc_, output));
    }
    if (use_relu_) {
      ReportCUDNNErrors(cudnnActivationForward(cudnn, activation_, &alpha,
                                               out_tensor_desc_, output, &beta,
                                               out_tensor_desc_, output));
    }
  }
#endif
}

template <typename DataType>
ConvLayer<DataType>::~ConvLayer() {
  ReportCUDAErrors(cudaFree(weights));
  ReportCUDAErrors(cudaFree(biases));

  cudnnDestroyFilterDescriptor(filter_desc_);
  cudnnDestroyConvolutionDescriptor(conv_desc_);
  cudnnDestroyTensorDescriptor(bias_desc_);
  cudnnDestroyTensorDescriptor(in_tensor_desc_);
  cudnnDestroyTensorDescriptor(out_tensor_desc_);
  cudnnDestroyActivationDescriptor(activation_);
}

template <typename DataType>
SoftMaxLayer<DataType>::SoftMaxLayer(BaseLayer<DataType>* ip)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip) {
  cudnnCreateTensorDescriptor(&out_tensor_desc_);
}

template <typename DataType>
SoftMaxLayer<DataType>::~SoftMaxLayer() {
  cudnnDestroyTensorDescriptor(out_tensor_desc_);
}

template <typename DataType>
void SoftMaxLayer<DataType>::Eval(int N, DataType* output,
                                  const DataType* input,
                                  const DataType* /*input2*/, void* /*scratch*/,
                                  size_t /*scratch_size*/, cudnnHandle_t cudnn,
                                  cublasHandle_t /*cublas*/) {
  float alpha = 1.0f, beta = 0.0f;

  const cudnnDataType_t dataType =
      std::is_same<half, DataType>::value ? CUDNN_DATA_HALF : CUDNN_DATA_FLOAT;

  const cudnnTensorFormat_t layout =
      nhwc_ ? CUDNN_TENSOR_NHWC : CUDNN_TENSOR_NCHW;

  // Need to call this at Eval as 'N' changes :-/
  cudnnSetTensor4dDescriptor(out_tensor_desc_, layout, dataType, N, GetC(),
                             GetH(), GetW());

  cudnnSoftmaxForward(cudnn, CUDNN_SOFTMAX_ACCURATE,
                      CUDNN_SOFTMAX_MODE_INSTANCE, &alpha, out_tensor_desc_,
                      input, &beta, out_tensor_desc_, output);
}


template <typename DataType>
SELayer<DataType>::SELayer(BaseLayer<DataType>* ip, int fc1Outputs,
                           bool addPrevLayerBias)
    : BaseLayer<DataType>(ip->GetC(), ip->GetH(), ip->GetW(), ip),
      numFc1Out_(fc1Outputs),
      addPrevLayerBias_(addPrevLayerBias) {
  ReportCUDAErrors(cudaMalloc(&w1_, C * numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&w2_, 2 * C * numFc1Out_ * sizeof(DataType)));

  if (kUseFusedSELayer && nhwc_) {
    ReportCUDAErrors(cudaMalloc(&w1_t_, C * numFc1Out_ * sizeof(DataType)));
    ReportCUDAErrors(cudaMalloc(&w2_t_, 2 * C * numFc1Out_ * sizeof(DataType)));
  }

  ReportCUDAErrors(cudaMalloc(&b1_, numFc1Out_ * sizeof(DataType)));
  ReportCUDAErrors(cudaMalloc(&b2_, 2 * C * sizeof(DataType)));

  ReportCUDAErrors(cudaMalloc(&bPrev_, C * sizeof(DataType)));
}

template <typename DataType>
SELayer<DataType>::~SELayer() {
  ReportCUDAErrors(cudaFree(w1_));
  ReportCUDAErrors(cudaFree(w2_));
  ReportCUDAErrors(cudaFree(b1_));
  ReportCUDAErrors(cudaFree(b2_));
  ReportCUDAErrors(cudaFree(bPrev_));
}

template <>
void SELayer<float>::LoadWeights(float* w1, float* b1, float* w2, float* b2,
                                 float* prevLayerBias, void* /*scratch*/) {
  const size_t num_weights1 = C * numFc1Out_;
  const size_t weight_size1 = sizeof(float) * num_weights1;

  const size_t weight_size2 = 2 * weight_size1;

  // Weight for the first FC layer.
  ReportCUDAErrors(cudaMemcpy(w1_, w1, weight_size1, cudaMemcpyHostToDevice));

  // Weight for the second FC layer.
  ReportCUDAErrors(cudaMemcpy(w2_, w2, weight_size2, cudaMemcpyHostToDevice));

  // Bias for the first FC layer.
  ReportCUDAErrors(
      cudaMemcpy(b1_, b1, numFc1Out_ * sizeof(float), cudaMemcpyHostToDevice));

  // Bias for the second FC layer.
  ReportCUDAErrors(
      cudaMemcpy(b2_, b2, 2 * C * sizeof(float), cudaMemcpyHostToDevice));

  // Bias for previous layer (Convolution).
  if (prevLayerBias) {
    ReportCUDAErrors(cudaMemcpy(bPrev_, prevLayerBias, C * sizeof(float),
                                cudaMemcpyHostToDevice));
  }
}


