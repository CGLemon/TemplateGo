#include "LZ/LZModel.h"
#include "blas/CPULayers.h"
#include "Winograd_helper.h"
#include "Utils.h"

using namespace Utils;

namespace LZ {

template <class container> void process_bn_var(container &weights) {
  constexpr float epsilon = 1e-5f;
  for (auto &&w : weights) {
    w = 1.0f / std::sqrt(w + epsilon);
  }
}

void Desc::ConvLayerDesc::loader_weights(std::vector<float> & weights) {
  m_weights = std::move(weights);
}

void Desc::ConvLayerDesc::loader_biases(std::vector<float> & biases) {
  m_biases = std::move(biases);
}

bool Desc::ConvLayerDesc::check() {
  const size_t w_size = oC * iC * W * H;
  const size_t b_size = oC;
  if (w_size == m_weights.size() &&
        b_size == m_biases.size()) {
    m_weights.reserve(w_size);
    m_biases.reserve(b_size);
    return true;
  } else {
    auto_printf("id=%zu : ConvLayer fail!\n", id);
    auto_printf("original size %zu and %zu: | object size %zu and %zu \n\n",
                m_weights.size(), m_biases.size(), w_size, b_size);
    return false;
  }
}

void Desc::BNLayerDesc::loader_means(std::vector<float> & means) {
  m_means = std::move(means);
}

void Desc::BNLayerDesc::loader_stddevs(std::vector<float> & stddevs) {
  process_bn_var(stddevs);
  m_stddevs = std::move(stddevs);
}

bool Desc::BNLayerDesc::check() {
  const size_t b_size = C;
  if (b_size == m_means.size() &&
        b_size == m_stddevs.size()) {
    m_means.reserve(b_size);
    m_stddevs.reserve(b_size);
    return true;
  } else {
    auto_printf("id=%zu : BNLayer fail!\n", id);
    auto_printf("original size %zu and %zu: | object size %zu and %zu \n\n",
                m_means.size(), m_stddevs.size(), b_size, b_size);
    return false;
  }
}


void Desc::FCLayerDesc::loader_weights(std::vector<float> & weights) {
  m_weights = std::move(weights);
}


void Desc::FCLayerDesc::loader_biases(std::vector<float> & biases) {
  m_biases = std::move(biases);
}

bool Desc::FCLayerDesc::check() {
  const size_t w_size = W;
  const size_t b_size = B;
  if (w_size == m_weights.size() &&
        b_size == m_biases.size()) {
    m_weights.reserve(w_size);
    m_biases.reserve(b_size);
    return true;
  } else {
    auto_printf("id=%zu : FCLayer fail!\n", id);
    auto_printf("original size %zu and %zu: | object size %zu and %zu \n\n",
                m_weights.size(), m_biases.size(), w_size, b_size);
    return false;
  }
}


bool Desc::ConvBlockDesc::check() {
  bool success = true;

  success &= m_conv.check();
  success &= m_batchnorm.check();

  return success;
}


bool Desc::ResidualDesc::check() {
  bool success = true;

  success &= m_conv_blocks[0].check();
  success &= m_conv_blocks[1].check();

  return success;
}

void LZModel::Loader::apply_convblock(Desc::ConvBlockDesc & conv, size_t & id,
                                      size_t oC, size_t iC, size_t W, size_t H) {
  auto weights = get_line();
  auto biases = get_line();
  auto means = get_line();
  auto stddevs = get_line();

  conv.m_conv.oC = oC;
  conv.m_conv.iC = iC;
  conv.m_conv.W = W;
  conv.m_conv.H = H;
  conv.m_conv.id = id;
  id++;

  conv.m_batchnorm.C = oC;
  conv.m_batchnorm.id = id;
  id++;

  conv.m_conv.loader_weights(weights);
  conv.m_conv.loader_biases(biases);
  conv.m_batchnorm.loader_means(means);
  conv.m_batchnorm.loader_stddevs(stddevs);
}


void LZModel::Loader::apply_fclayer(Desc::FCLayerDesc & fc, size_t & id,
                                    size_t intputs, size_t outputs) {
  auto weights = get_line();
  auto biases = get_line();
  fc.id = id;
  fc.W = intputs * outputs;
  fc.B = outputs;
  id++;

  fc.loader_weights(weights);
  fc.loader_biases(biases);
}

void LZModel::Loader::apply_resblock(Desc::ResidualDesc & resblock, size_t & id,
                                     size_t oC, size_t iC, size_t W, size_t H) {
  apply_convblock(resblock.m_conv_blocks[0], id, oC, iC, W, H);
  apply_convblock(resblock.m_conv_blocks[1], id, oC, iC, W, H);

  assert(iC == oC);
  resblock.m_num_channels = iC;
}

bool LZModel::Loader::is_end() {
  auto line = std::string{};
  if (std::getline(m_weight_str, line)) {
    return false;
  } 
  return true;

}


std::vector<float> LZModel::Loader::get_line() {
  auto res = std::vector<float>{};
  auto line = std::string{};

  if (std::getline(m_weight_str, line)) {
    float weight;
    std::istringstream iss(line);
    while (iss >> weight) {
      res.emplace_back(weight);
    }
  } else {
    auto_printf("Should not be happened?");
    exit(-1);
  }
  return res;
}


bool LZModel::ForwardPipeWeights::check() {

  bool success = true;

  success &= m_ip_conv.check();

  for (auto& res : m_res_blocks) {
    success &= res.check();
  }

  success &= m_conv_pol.check();
  success &= m_fc_pol.check();
  success &= m_conv_val.check();
  success &= m_fc1_val.check();
  success &= m_fc2_val.check();

  return success;
}


void LZModel::loader(const std::string &filename, 
                     std::shared_ptr<LZModel::ForwardPipeWeights> weights) {
  auto buffer = std::stringstream{};

#ifdef USE_ZLIB
  auto gzhandle = gzopen(filename.c_str(), "rb");
  if (gzhandle == nullptr) {
    auto_printf("Could not open weights file: %s\n", filename.c_str());
  }

  constexpr auto chunkBufferSize = 64 * 1024;
  std::vector<char> chunkBuffer(chunkBufferSize);

  while (true) {
    auto bytesRead = gzread(gzhandle, chunkBuffer.data(), chunkBufferSize);
    if (bytesRead == 0)
      break;
    if (bytesRead < 0) {
      auto_printf("Failed to decompress or read: %s\n", filename.c_str());
      gzclose(gzhandle);
    }
    assert(bytesRead <= chunkBufferSize);
    buffer.write(chunkBuffer.data(), bytesRead);
  }
  gzclose(gzhandle);
#else
  std::ifstream weights_file(filename.c_str());
  auto stream_line = std::string{};
  while (std::getline(weights_file, stream_line)) {
    buffer << stream_line << std::endl;
  }
  weights_file.close();
#endif
  

  auto line = std::string{};
  
  int format_version = -1;
  if (std::getline(buffer, line)) {
    auto iss = std::stringstream{line};
    iss >> format_version;
    if (iss.fail() || (format_version != 1 && format_version != 2)) {
      auto_printf("Weights file is the wrong version.\n");
    } else {
      if (format_version == 2) {
        auto_printf("ELF OpenGO network file is not suport.\n");
      } else {
        auto_printf("Loading Leelaz network file.\n");
      }
      fill_weights(buffer, weights);
    }
  }
  
}

void LZModel::fill_weights(std::istream &wtfile,
                           std::shared_ptr<LZModel::ForwardPipeWeights> pipe_weights) {
  auto_printf("Detecting residual layers...");
  
  std::vector<float> weights;
  int linecount = 1;
  int channels = 0;
  auto line = std::string{};
  while (std::getline(wtfile, line)) {
    auto iss = std::stringstream{line};
    if (linecount == 2) {
      auto count = std::distance(std::istream_iterator<std::string>(iss),
                                 std::istream_iterator<std::string>());
      auto_printf("%d channels...", count);
      channels = count;
    }
    linecount++;
  }
  const int version_lines = 1;
  const int tower_head_conv_lines = 4;
  const int linear_lines = 14;
  const int not_residual_lines =
      version_lines + tower_head_conv_lines + linear_lines;
  const int residual_lines = linecount - not_residual_lines;
  if (residual_lines % 8 != 0) {
    auto_printf("\nInconsistent number of weights in the file.\n");
  }

  const int residual_blocks = residual_lines / 8;
  auto_printf("%d blocks.\n", residual_blocks);

  wtfile.clear();
  wtfile.seekg(0, std::ios::beg);

  std::getline(wtfile, line);
  
  Loader loader(wtfile);
  size_t id = 0;
  const size_t filter_3 = RESIDUAL_FILTER;
  const size_t filter_1 = HEAD_FILTER;
  
  loader.apply_convblock(pipe_weights->m_ip_conv, id, channels, LZ::INPUT_CHANNELS, filter_3, filter_3);
  for (size_t i = 0; i < residual_blocks; ++i) {
    pipe_weights->m_res_blocks.emplace_back(Desc::ResidualDesc{});
    loader.apply_resblock(pipe_weights->m_res_blocks[i], id, channels, channels, filter_3, filter_3);
  }
  loader.apply_convblock(pipe_weights->m_conv_pol, id, LZ::OUTPUTS_POLICY, channels, filter_1, filter_1);

  const size_t plane_policy = NUM_INTERSECTIONS * LZ::OUTPUTS_POLICY;
  loader.apply_fclayer(pipe_weights->m_fc_pol, id, plane_policy, LZ::POTENTIAL_MOVES);

  loader.apply_convblock(pipe_weights->m_conv_val, id, LZ::OUTPUTS_VALUE, channels, filter_1, filter_1);

  const size_t plane_value = NUM_INTERSECTIONS * LZ::OUTPUTS_VALUE;
  loader.apply_fclayer(pipe_weights->m_fc1_val, id, plane_value, LZ::VALUE_LAYER);
  loader.apply_fclayer(pipe_weights->m_fc2_val, id, LZ::VALUE_LAYER, LZ::VALUE_LABELS);
  
  bool success = true;
  success &= pipe_weights->check();
  success &= loader.is_end();
  assert(success);
}

void LZModel::transform(bool is_winograd,
                        std::shared_ptr<LZModel::ForwardPipeWeights> weights) {

  const size_t residual_channels = weights->get_num_channels(0);
  const size_t residual_blocks = weights->get_num_residuals();


  for (auto i = size_t{0}; i < residual_channels; i++) {
    weights->m_ip_conv.m_batchnorm.m_means[i] -=
        weights->m_ip_conv.m_conv.m_biases[i];
    weights->m_ip_conv.m_conv.m_biases[i] = 0.0f;
  }

  for (auto i = size_t{0}; i < residual_blocks; i++) {
    auto means_size = residual_channels;
    for (auto j = size_t{0}; j < means_size; j++) {
      weights->m_res_blocks[i].m_conv_blocks[0].m_batchnorm.m_means[j] -=
          weights->m_res_blocks[i].m_conv_blocks[0].m_conv.m_biases[j];
      weights->m_res_blocks[i].m_conv_blocks[0].m_conv.m_biases[j] = 0.0f;

      weights->m_res_blocks[i].m_conv_blocks[1].m_batchnorm.m_means[j] -=
          weights->m_res_blocks[i].m_conv_blocks[1].m_conv.m_biases[j];
      weights->m_res_blocks[i].m_conv_blocks[1].m_conv.m_biases[j] = 0.0f;
    }
  }

  for (auto i = size_t{0}; i < LZ::OUTPUTS_POLICY; i++) {
    weights->m_conv_pol.m_batchnorm.m_means[i] -=
        weights->m_conv_pol.m_conv.m_biases[i];
    weights->m_conv_pol.m_conv.m_biases[i] = 0.0f;
  }

  for (auto i = size_t{0}; i < LZ::OUTPUTS_VALUE; i++) {
    weights->m_conv_val.m_batchnorm.m_means[i] -=
        weights->m_conv_val.m_conv.m_biases[i];
    weights->m_conv_val.m_conv.m_biases[i] = 0.0f;
  }

  if (is_winograd) {
    weights->m_ip_conv.m_conv.m_weights = winograd_transform_f(
        weights->m_ip_conv.m_conv.m_weights, residual_channels, LZ::INPUT_CHANNELS);

    for (auto i = size_t{0}; i < residual_blocks; i++) {
      auto convweights_1 = weights->m_res_blocks[i].m_conv_blocks[0].m_conv.m_weights;
      auto convweights_2 = weights->m_res_blocks[i].m_conv_blocks[1].m_conv.m_weights;
      weights->m_res_blocks[i].m_conv_blocks[0].m_conv.m_weights 
             = winograd_transform_f(convweights_1, residual_channels, residual_channels); 
      weights->m_res_blocks[i].m_conv_blocks[1].m_conv.m_weights 
             = winograd_transform_f(convweights_2, residual_channels, residual_channels); 
    }
  }
}

std::pair<int, int> get_intersections_pair(int idx, int boardsize) {
  const int x = idx % boardsize;
  const int y = idx / boardsize;
  return {x, y};
}

void fill_input_plane_pair(const std::shared_ptr<Board> board,
                                    std::vector<float>::iterator black,
                                    std::vector<float>::iterator white,
                                    const int symmetry) {
  for (int idx = 0; idx < NUM_INTERSECTIONS; idx++) {
    const int sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
    const auto sym_pair = get_intersections_pair(sym_idx, BOARD_SIZE);
    const int x = sym_pair.first;
    const int y = sym_pair.second;
    const int color = board->get_state(x, y);
    if (color == Board::BLACK) {
      black[idx] = static_cast<float>(true);
    } else if (color == Board::WHITE) {
      white[idx] = static_cast<float>(true);
    }
  }
}


std::vector<float> LZModel::gather_features(const GameState *const state, const int symmetry) {
 
  auto input_data = std::vector<float>(LZ::INPUT_CHANNELS * NUM_INTERSECTIONS, 0.0f);

  const auto to_move = state->board.get_to_move();
  const auto blacks_move = to_move == Board::BLACK;

  const auto black_it =
      blacks_move ? begin(input_data)
                  : begin(input_data) + LZ::INPUT_MOVES * NUM_INTERSECTIONS;
  const auto white_it =
      blacks_move ? begin(input_data) + LZ::INPUT_MOVES * NUM_INTERSECTIONS
                  : begin(input_data);
  const auto to_move_it =
      blacks_move
          ? begin(input_data) + 2 * LZ::INPUT_MOVES * NUM_INTERSECTIONS
          : begin(input_data) + (2 * LZ::INPUT_MOVES + 1) * NUM_INTERSECTIONS;

  const auto moves =
      std::min<size_t>(state->board.get_movenum() + 1, LZ::INPUT_MOVES);
  // Go back in time, fill history boards

  for (auto h = size_t{0}; h < moves; h++) {
    // collect white, black occupation planes
    fill_input_plane_pair(state->get_past_board(h),
                          black_it + h * NUM_INTERSECTIONS,
                          white_it + h * NUM_INTERSECTIONS, symmetry);
  }
  std::fill(to_move_it, to_move_it + NUM_INTERSECTIONS, static_cast<float>(true));

  return input_data;
}


NNResult LZModel::get_result(std::vector<float> & policy,
                             std::vector<float> & value,
                             const float softmax_temp,
                             const int symmetry) {
  NNResult result;
  const auto outputs = Activation::softmax(policy, softmax_temp);

  // Map TanH output range [-1..1] to [0..1] range
  const auto winrate = (1.0f + std::tanh(value[0])) / 2.0f;

  for (auto idx = size_t{0}; idx < NUM_INTERSECTIONS; idx++) {
    const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
    result.policy[sym_idx] = outputs[idx];
  }

  result.policy_pass = outputs[NUM_INTERSECTIONS];
  result.winrate[0] = winrate;

  return result;
}

}
