#include "Model.h"
#include "Utils.h"
#include "Winograd_helper.h"
#include "Random.h"
#include "config.h"

#include <iterator>
#include <iomanip>
#include <functional>

using namespace Utils;

template <class container>
void process_bn_var(container &weights) {
  constexpr float epsilon = 1e-5f;
  for (auto &&w : weights) {
    w = 1.0f / std::sqrt(w + epsilon);
  }
}

void Desc::ConvLayer::load_weights(std::vector<float> &loadweights) {
  weights = std::move(loadweights);
}

void Desc::BatchNormLayer::load_means(std::vector<float> &loadweights) {
  means = std::move(loadweights);
}

void Desc::BatchNormLayer::load_stddevs(std::vector<float> &loadweights) {
  process_bn_var(loadweights);
  stddevs = std::move(loadweights);
}

void Desc::LinearLayer::load_weights(std::vector<float> &loadweights) {
  weights = std::move(loadweights);
}

void Desc::LinearLayer::load_biases(std::vector<float> &loadweights) {
  biases = std::move(loadweights);
}

void Model::loader(const std::string &filename,
                   std::shared_ptr<NNweights> &nn_weight) {
  std::ifstream file;
  std::stringstream buffer;
  std::string line;

  file.open(filename.c_str());

  if (!file.is_open()) {
    file.close();
    static_printf("Could not opne file : %s\n", filename.c_str());
    return;
  }

  while(std::getline(file, line)) {
    buffer << line;
    buffer << std::endl;
  }
  file.close();
  fill_weights(buffer, nn_weight);
}

std::vector<float> get_weights_from_file(std::istream &weights_file) {
  auto weights = std::vector<float>{};
  auto line = std::string{};

  if (std::getline(weights_file, line)) {
    float weight;
    std::stringstream line_buffer(line);
    while(line_buffer >> weight) {
      weights.emplace_back(weight);
    }
  }
  return weights;
}

void Model::fill_weights(std::istream &weights_file,
                         std::shared_ptr<NNweights> &nn_weight) {

  size_t line_cnt = 0;
  size_t channels = 0;
  
  auto line = std::string{};
  while (std::getline(weights_file, line)) {

    auto iss = std::stringstream{line};
    if (line_cnt == 2) {
      auto count = std::distance(std::istream_iterator<std::string>(iss),
                                 std::istream_iterator<std::string>());
      channels = count;
    }
    line_cnt++;
  }

  weights_file.clear();
  weights_file.seekg(0, std::ios::beg);
  auto weights = std::vector<float>{};

  const size_t input_cnt = 5;
  const size_t single_tower_lines_cnt = 10;
  const size_t head_cnt = 15;

  const size_t res_cnt = line_cnt - head_cnt - input_cnt;
  const size_t residuals = res_cnt / single_tower_lines_cnt;

  assert(res_cnt % single_tower_lines_cnt == 0);

  nn_weight->channels = channels;
  nn_weight->residuals = residuals;

  auto_printf("%zu channels\n", channels);
  auto_printf("%zu residuals\n", residuals);


  // input
  fill_convolution_layer(nn_weight->input_conv, weights_file);

  fill_batchnorm_layer(nn_weight->input_bn, weights_file);

  fill_fullyconnect_layer(nn_weight->input_fc, weights_file);

  // residual tower
  for (auto b = size_t{0}; b < residuals; ++b) {
    nn_weight->residual_tower.emplace_back(NNweights::ResidualTower{});
    auto tower_ptr = nn_weight->residual_tower.data() + b;
    
    fill_convolution_layer(tower_ptr->conv_1, weights_file);
    fill_batchnorm_layer(tower_ptr->bn_1, weights_file);

    fill_convolution_layer(tower_ptr->conv_2, weights_file);
    fill_batchnorm_layer(tower_ptr->bn_2, weights_file);


    fill_fullyconnect_layer(tower_ptr->extend, weights_file);
    fill_fullyconnect_layer(tower_ptr->squeeze, weights_file);
  }
  // policy head
  fill_convolution_layer(nn_weight->p_conv, weights_file);
  fill_batchnorm_layer(nn_weight->p_bn, weights_file);

  fill_convolution_layer(nn_weight->prob_conv, weights_file);
  fill_fullyconnect_layer(nn_weight->pass_fc, weights_file);

  // value head
  fill_convolution_layer(nn_weight->v_conv, weights_file);
  fill_batchnorm_layer(nn_weight->v_bn, weights_file);
 
  fill_convolution_layer(nn_weight->sb_conv, weights_file);
  fill_convolution_layer(nn_weight->os_conv, weights_file);
  fill_fullyconnect_layer(nn_weight->fs_fc, weights_file);
  fill_fullyconnect_layer(nn_weight->v_fc, weights_file);

  weights = get_weights_from_file(weights_file);
  assert(weights.size() == 0);

  nn_weight->loaded = true;
}

std::pair<int, int> get_intersections_pair(int idx, int boardsize) {
  const int x = idx % boardsize;
  const int y = idx / boardsize;
  return {x, y};
}


void fill_color_plane_pair(const std::shared_ptr<Board> board,
                           std::vector<float>::iterator black,
                           std::vector<float>::iterator white,
                           const int symmetry) {

  const int boardsize = board->get_boardsize();
  const int intersections = board->get_intersections();

  for (auto idx = size_t{0}; idx < intersections; ++idx) {
    const int sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
    const int x = sym_idx % boardsize;
    const int y = sym_idx / boardsize;
    const int vtx = board->get_vertex(x, y);
    const int color = board->get_state(vtx);
    if (color == Board::BLACK) {
      black[idx] = static_cast<float>(true);
    } else if (color == Board::WHITE) {
      white[idx] = static_cast<float>(true);
    }
  }
}

void fill_libs_plane_pair(const std::shared_ptr<Board> board,
                          std::vector<float>::iterator black,
                          std::vector<float>::iterator white,
                          const int condition,
                          const int symmetry) {

  const int boardsize = board->get_boardsize();
  const int intersections = board->get_intersections();

  for (auto idx = size_t{0}; idx < intersections; ++idx) {
    const int sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
    const int x = sym_idx % boardsize;
    const int y = sym_idx / boardsize;
    const int vtx = board->get_vertex(x, y);
    const int libs = board->get_libs(vtx);
    const int color = board->get_state(vtx);
    if (color == Board::BLACK && libs == condition) {
      black[idx] = static_cast<float>(true);
    } else if (color == Board::WHITE && libs == condition) {
      white[idx] = static_cast<float>(true);
    }
  }
}

void fill_ko_plane(const std::shared_ptr<Board> board,
                   std::vector<float>::iterator plane,
                   const int symmetry) {

  const int komove = board->get_komove();
  if (komove == Board::NO_VERTEX) {
    return;
  }
  const int x = board->get_x(komove);
  const int y = board->get_y(komove);
  const int idx = board->get_index(x, y);
  const int sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
  plane[sym_idx] = static_cast<float>(true);
}

void fill_move_plane(const std::shared_ptr<Board> board,
                     std::vector<float>::iterator plane,
                     const int symmetry) {

  const int last_move = board->get_last_move();
  if (last_move == Board::NO_VERTEX || last_move == Board::PASS || last_move == Board::RESIGN) {
    return;
  }
  const int x = board->get_x(last_move);
  const int y = board->get_y(last_move);
  const int idx = board->get_index(x, y);
  const int sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
  plane[sym_idx] = static_cast<float>(true);

}


std::vector<float> Model::gather_planes(const GameState *const state, 
                                        const int symmetry) {
  static constexpr auto PAST_MOVES = 3;
  static constexpr auto INPUT_PAIRS = 6;

  const int intersections = state->board.get_intersections();

  auto input_data = std::vector<float>(INPUT_CHANNELS * intersections, 0.0f);

  const auto to_move = state->board.get_to_move();
  const auto blacks_move = to_move == Board::BLACK;

  auto iterate = std::begin(input_data);
  auto black_it = blacks_move ? iterate
                              : iterate + INPUT_PAIRS * intersections;
  auto white_it = blacks_move ? iterate + INPUT_PAIRS * intersections
                              : iterate;

  const auto moves =
      std::min<size_t>(state->board.get_movenum() + 1, PAST_MOVES);

  for (auto h = size_t{0}; h < moves; ++h) {
    fill_color_plane_pair(state->get_past_board(h),
                          black_it + h * intersections,
                          white_it + h * intersections,
                          symmetry);
  }
  black_it += PAST_MOVES * intersections;
  white_it += PAST_MOVES * intersections;

  for (int c = 0; c < 3; ++c) {
    fill_libs_plane_pair(state->get_past_board(0),
                       black_it + c * intersections,
                       white_it + c * intersections,
                       c+1, symmetry);
  }
  iterate += 2 * INPUT_PAIRS * intersections;


  fill_ko_plane(state->get_past_board(0),
                iterate, symmetry);
  iterate += intersections;

  for (auto h = size_t{0}; h < moves; ++h) {
    fill_move_plane(state->get_past_board(h),
                    iterate + h * intersections,
                    symmetry);
  }
  iterate += PAST_MOVES * intersections;

  if (blacks_move) {
    std::fill(iterate, iterate+intersections, static_cast<float>(true));
  }
  iterate += intersections;


  std::fill(iterate, iterate+intersections, static_cast<float>(true));

  return input_data;
}

std::vector<float> Model::gather_features(const GameState *const state) {

  static constexpr auto FEATURE_PASS = 5;
  static constexpr auto FEATURE_KO = 3;

  auto input_data = std::vector<float>(INPUT_FEATURES, 0.0f);
  
  auto roll = size_t{0};
  const auto moves =
      std::min<size_t>(state->board.get_movenum() + 1, FEATURE_PASS);

  for (auto i = size_t{0}; i < moves; ++i) {
    const auto board = state->get_past_board(i);
    const auto move = board->get_last_move();
    if (move == Board::PASS) { 
      input_data[roll + i] = 1.0f;
    }
  }
  roll += FEATURE_PASS;

  const auto komi = state->board.get_komi();
  input_data[roll] = komi / 15.0f;
  
  const auto boardsize = state->board.get_boardsize();
  input_data[roll + 1] = boardsize / 5.0f;
  roll += 2;

  const auto ko_past =
      std::min<size_t>(state->board.get_movenum() + 1, FEATURE_KO);

  for (auto i = size_t{0}; i < ko_past; ++i) {
    const auto board = state->get_past_board(i);
    const auto ko_move = board->get_komove();
    if (ko_move != Board::NO_VERTEX) { 
      input_data[roll + i] = 1.0f;
    }
  }
  return input_data;
}

void Model::features_stream(std::ostream &out, const GameState *const state) {

  const auto planes = gather_planes(state, Board::IDENTITY_SYMMETRY);
  const auto features = gather_features(state);
  const auto boardsize = state->board.get_boardsize();
  const auto intersections = boardsize * boardsize;
  const auto input_channels = INPUT_CHANNELS;
  const auto input_features = INPUT_FEATURES;

  for (auto i = size_t{0}; i < input_channels; ++i) {
    out << std::endl;
    out << "patten : " << (i+1) << std::endl;
    for (auto idx = size_t{0}; idx < intersections; ++idx) {
      out << std::setw(4) << planes[idx + i * intersections] << " ";
      if (idx % boardsize == (boardsize-1)) {
        out << std::endl;
      }
    }
  }

  out << std::endl;
  out << "features" << std::endl;
  for (auto i = size_t{0}; i < input_features; ++i) {
    out << std::setw(4) << features[i];
    out << std::endl;
  }
}

std::string Model::features_to_string(GameState &state) {

  auto out = std::ostringstream{};
  features_stream(out, &state);

  return out.str();
}

NNResult Model::get_result(const GameState *const state,
                           std::vector<float> &policy,
                           std::vector<float> &score_belief,
                           std::vector<float> &ownership,
                           std::vector<float> &final_score,
                           std::vector<float> &value,
                           const float softmax_temp,
                           const int symmetry) {
  NNResult result;

  const auto boardsize = state->board.get_boardsize();
  const auto intersections = boardsize * boardsize;

  // Probabilities
  const auto probabilities = Activation::Softmax(policy, softmax_temp);
  for (auto idx = size_t{0}; idx < intersections; ++idx) {
    const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
    result.policy[sym_idx] = probabilities[idx];
  }
  result.policy_pass = probabilities[intersections];

  // Score belief
  const auto score = Activation::Softmax(score_belief, softmax_temp);
  for (auto idx = size_t{0}; idx < OUTPUTS_SCOREBELIEF * intersections; ++idx) {
    result.score_belief[idx] = score[idx];
  }
  
  // Ownership
  for (auto idx = size_t{0}; idx < intersections; ++idx) {
    const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
    result.ownership[sym_idx] = std::tanh(ownership[idx]);
  }

  // Final score
  result.final_score = 20 * final_score[0];

  // Map TanH output range [-1..1] to [0..1] range
  for (auto idx = size_t{0}; idx < VALUE_LABELS; ++idx) {
    const auto winrate = (1.0f + std::tanh(value[idx])) / 2.0f;
    result.multi_labeled[idx] = winrate;
  }

  int label_choice = LABELS_CENTER + cfg_lable_komi + cfg_lable_shift;
  if (label_choice < 0) {
    label_choice = 0;
  } else if (label_choice >= VALUE_LABELS) {
    label_choice = VALUE_LABELS - 1;
  }

  result.winrate = result.multi_labeled[label_choice];

  return result;
}

NNResult Model::get_result_form_cache(NNResult result) {
  // 依據當前的動態貼目選擇新的勝率
  int label_choice = LABELS_CENTER + cfg_lable_komi + cfg_lable_shift;
  if (label_choice < 0) {
    label_choice = 0;
  } else if (label_choice >= VALUE_LABELS) {
    label_choice = VALUE_LABELS - 1;
  }

  result.winrate = result.multi_labeled[label_choice];

  return result;
}

void Model::winograd_transform(std::shared_ptr<NNweights> &nn_weight) {

  auto channels = nn_weight->channels;
  nn_weight->input_conv.weights = winograd_transform_f(
      nn_weight->input_conv.weights, channels, INPUT_CHANNELS);

  for (auto &tower_ref : nn_weight->residual_tower) {
     tower_ref.conv_1.weights = winograd_transform_f(
        tower_ref.conv_1.weights, channels, channels);

     tower_ref.conv_2.weights = winograd_transform_f(
        tower_ref.conv_2.weights, channels, channels);
  }
}

void Model::fill_fullyconnect_layer(Desc::LinearLayer &layer, std::istream &weights_file) {
  auto weights = get_weights_from_file(weights_file);
  layer.load_weights(weights);

  weights = get_weights_from_file(weights_file);
  layer.load_biases(weights);
}

void Model::fill_batchnorm_layer(Desc::BatchNormLayer &layer, std::istream &weights_file) {
  auto  weights = get_weights_from_file(weights_file);
  layer.load_means(weights);

  weights = get_weights_from_file(weights_file);
  layer.load_stddevs(weights);
}

void Model::fill_convolution_layer(Desc::ConvLayer &layer, std::istream &weights_file) {
  auto weights = get_weights_from_file(weights_file);
  layer.load_weights(weights);
}

