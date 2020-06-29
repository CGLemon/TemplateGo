#ifndef LZMODEL_H_INCLUDE
#define LZMODEL_H_INCLUDE

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifdef USE_ZLIB
#include "zlib.h"
#endif

#include "GameState.h"
#include "Board.h"
#include "LZ/LZNetParameters.h"
#include "config.h"
#include "CacheTable.h"

namespace LZ {

class Desc {
public:
  struct ConvLayerDesc {
    void loader_weights(std::vector<float> & weights);
    void loader_biases(std::vector<float> & biases);

    size_t N, C, W, H, id;
    std::vector<float> m_weights;
    std::vector<float> m_biases;
    
    bool check();
  };

  struct BNLayerDesc {
    void loader_means(std::vector<float> & means);
    void loader_stddevs(std::vector<float> & stddevs);

    size_t C, id;
    std::vector<float> m_means;
    std::vector<float> m_stddevs;
    bool check();
  };

  struct FCLayerDesc {
    void loader_weights(std::vector<float> & weights);
    void loader_biases(std::vector<float> & biases);

    size_t W, B, id;
    std::vector<float> m_weights;
    std::vector<float> m_biases;
    bool check();
  };
 
  struct ConvBlockDesc {
    ConvLayerDesc m_conv;
    BNLayerDesc m_batchnorm;
    bool check();
  };

  struct ResidualDesc {
    std::array<ConvBlockDesc, 2> m_conv_blocks;
    size_t m_num_channels{0};
    bool check();
  };

};


class LZModel {
public:
  struct ForwardPipeWeights { 
    // Input layer
    Desc::ConvBlockDesc m_ip_conv;

    // Residual tower
    std::vector<Desc::ResidualDesc> m_res_blocks;

    // Policy head
    Desc::ConvBlockDesc m_conv_pol;
    Desc::FCLayerDesc m_fc_pol;

    // Value head
    Desc::ConvBlockDesc m_conv_val;
    Desc::FCLayerDesc m_fc1_val;
    Desc::FCLayerDesc m_fc2_val;

    size_t get_num_residuals() { return m_res_blocks.size(); }
    size_t get_num_channels(int id) { 
      if (id > m_res_blocks.size()-1) { return 0; } 
      else { return m_res_blocks[id].m_num_channels; }
    }
    bool check();
  };

  class Loader {
    public:
      Loader(std::istream & wtfile) : m_weight_str(wtfile) {};

      void apply_convblock(Desc::ConvBlockDesc & conv, size_t & id,
                           size_t N, size_t C, size_t W, size_t H);

      void apply_resblock(Desc::ResidualDesc & resblock, size_t & id,
                          size_t N, size_t C, size_t W, size_t H);

      void apply_fclayer(Desc::FCLayerDesc & fc, size_t & id,
                         size_t inputs, size_t outputs);

      bool is_end();

    private:
      std::vector<float> get_line();
      std::istream & m_weight_str;
  };

  static void loader(const std::string &filename, 
                     std::shared_ptr<LZModel::ForwardPipeWeights> weights);
  static void fill_weights(std::istream &wtfile,
                           std::shared_ptr<LZModel::ForwardPipeWeights> weights);

  static void transform(bool is_winograd,
                        std::shared_ptr<LZModel::ForwardPipeWeights> weights);

  static std::vector<float> gather_features(const GameState *const state, 
                                            const int symmetry);

  static NNResult get_result(std::vector<float> & policy,
                             std::vector<float> & value,
                             const float softmax_temp,
                             const int symmetry);
};


class NNpipe {
public:
  virtual void initialize(std::shared_ptr<LZModel::ForwardPipeWeights> weights) = 0;
  virtual void forward(const std::vector<float> &input,
                       std::vector<float> &output_pol,
                       std::vector<float> &output_val) = 0;

};

}
#endif
