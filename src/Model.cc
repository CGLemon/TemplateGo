#include "Model.h"
#include "Utils.h"
#include "Winograd_helper.h"
#include "Random.h"
#include "config.h"

#include <iterator>
#include <iomanip>
#include <functional>

using namespace Utils;

template <typename container>
void process_bn_var(container &weights) {
    static constexpr float epsilon = 1e-5f;
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
        auto_printf("Could not opne file : %s\n", filename.c_str());
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
        // On MacOS, if the number is too small, stringstream
        // can not parser the number to float.
        double weight;
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

void fill_color_plane_pair(const std::shared_ptr<Board> board,
                           std::vector<float>::iterator black,
                           std::vector<float>::iterator white,
                           const int symmetry) {

    const auto boardsize = board->get_boardsize();
    const auto intersections = board->get_intersections();

    for (int idx = 0; idx < intersections; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % boardsize;
        const auto y = sym_idx / boardsize;
        const auto vtx = board->get_vertex(x, y);
        const auto color = board->get_state(vtx);
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

    const auto boardsize = board->get_boardsize();
    const auto intersections = board->get_intersections();

    for (int idx = 0; idx < intersections; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % boardsize;
        const auto y = sym_idx / boardsize;
        const auto vtx = board->get_vertex(x, y);
        const auto libs = board->get_libs(vtx);
        const auto color = board->get_state(vtx);
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

    const auto komove = board->get_komove();
    if (komove == Board::NO_VERTEX) {
        return;
    }
    const auto intersections = board->get_intersections();
    const auto x = board->get_x(komove);
    const auto y = board->get_y(komove);
    const auto ko_idx = board->get_index(x, y);

    for (int idx = 0; idx < intersections; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        if (ko_idx == sym_idx) {
            plane[idx] = static_cast<float>(true);
            break;
        }
    }
}

void fill_move_plane(const std::shared_ptr<Board> board,
                     std::vector<float>::iterator plane,
                     const int symmetry) {

    const int last_move = board->get_last_move();
    if (last_move == Board::NO_VERTEX || last_move == Board::PASS || last_move == Board::RESIGN) {
        return;
    }
    const auto intersections = board->get_intersections();
    const int x = board->get_x(last_move);
    const int y = board->get_y(last_move);
    const int lastmove_idx = board->get_index(x, y);

    for (int idx = 0; idx < intersections; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        if (lastmove_idx == sym_idx) {
            plane[idx] = static_cast<float>(true);
            break;
        }
    }
}

void fill_seki_plane(const std::shared_ptr<Board> board,
                     std::vector<float>::iterator plane,
                     const int symmetry) {

    const auto ownership = board->get_ownership();
    const auto intersections = board->get_intersections();

    for (int idx = 0; idx < intersections; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        if (ownership[sym_idx] == Board::EMPTY) {
            plane[idx] = static_cast<float>(true);
        }
    }
}

void fill_takemove_plane(const std::shared_ptr<Board> board,
                         std::vector<float>::iterator plane,
                         const int symmetry) {

    const auto boardsize = board->get_boardsize();
    const auto intersections = board->get_intersections();

    for (int idx = 0; idx < intersections; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        const auto x = sym_idx % boardsize;
        const auto y = sym_idx / boardsize;

        const auto vtx = board->get_vertex(x, y);
        const auto color = board->get_state(vtx);
        const auto take = board->is_take_move(vtx, color) ||
                              board->is_take_move(vtx, !color);

        if (take) {
            plane[idx] = static_cast<float>(true);
        }
    }
}


void fill_ladder_plane(const std::shared_ptr<Board> board,
                       std::vector<float>::iterator plane,
                       const int symmetry) {

    const auto ladders = board->get_ladders();
    const auto intersections = board->get_intersections();

    for (int idx = 0; idx < intersections; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        if (ladders[sym_idx] == Board::ladder_t::LADDER_DEATH) {
            plane[idx + 0 * intersections] = static_cast<float>(true);
        } 
        else if (ladders[sym_idx] == Board::ladder_t::LADDER_ESCAPABLE) {
            plane[idx + 1 * intersections] = static_cast<float>(true);
        }
        else if (ladders[sym_idx] == Board::ladder_t::LADDER_ATARI) {
            plane[idx + 2 * intersections] = static_cast<float>(true);
        }
        else if (ladders[sym_idx] == Board::ladder_t::LADDER_TAKE) {
            plane[idx + 3 * intersections] = static_cast<float>(true);
        }
    }
}

std::vector<float> Model::gather_planes(const GameState *const state, 
                                        const int symmetry) {
    static constexpr auto PAST_MOVES = 3;
    static constexpr auto INPUT_PAIRS = 6;
    static constexpr auto LADDERS = 4;
   /*
    * 
    * Plane  1: current player now stones on board
    * Plane  2: current player one past move stones on board
    * Plane  3: current player two past move stones on board
    *
    * Plane  4: current player one liberty strings
    * Plane  5: current player two liberties strings
    * Plane  6: current player three liberties strings
    *
    * Plane  7: next player now stones on board
    * Plane  8: next player one past move stones on board
    * Plane  9: next player two past move stones on board
    *
    * Plane 10: current player one liberty strings
    * Plane 11: current player two liberties strings
    * Plane 12: current player three liberties strings
    *
    * Plane 13: ko move (one hot)
    *
    * Plane 14: now move (one hot)
    * Plane 15: one past move (one hot)
    * Plane 16: two past move (one hot)
    *
    * Plane 17: seki point
    * Plane 18: take move
    *
    * Plane 19: ladder features
    * Plane 20: ladder features
    * Plane 21: ladder features
    * Plane 22: ladder features
    *
    * Plane 23: black or white
    * Plane 24: always one
    *
    */

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
    // plane 1 to 3 and plane 7 to 9  
    for (auto h = size_t{0}; h < moves; ++h) {
        fill_color_plane_pair(state->get_past_board(h),
                              black_it + h * intersections,
                              white_it + h * intersections,
                              symmetry);
    }
    black_it += PAST_MOVES * intersections;
    white_it += PAST_MOVES * intersections;

    // plane 4 to 6 and plane 10 to 12  
    for (int c = 0; c < 3; ++c) {
        fill_libs_plane_pair(state->get_past_board(0),
                             black_it + c * intersections,
                             white_it + c * intersections,
                             c+1, symmetry);
    }
    std::advance(iterate, 2 * INPUT_PAIRS * intersections);

    // plane 13
    fill_ko_plane(state->get_past_board(0),
                  iterate, symmetry);
    std::advance(iterate, intersections);

    // plane 14 to 16
    for (auto h = size_t{0}; h < moves; ++h) {
        fill_move_plane(state->get_past_board(h),
                        iterate + h * intersections,
                        symmetry);
    }
    std::advance(iterate, PAST_MOVES * intersections);

    // plane 17
    fill_seki_plane(state->get_past_board(0),
                    iterate,
                    symmetry);
    std::advance(iterate,  intersections);

    // plane 18
    fill_takemove_plane(state->get_past_board(0),
                        iterate,
                        symmetry);
    std::advance(iterate,  intersections);


    // plane 19 to 22
    fill_ladder_plane(state->get_past_board(0),
                      iterate,
                      symmetry);
    std::advance(iterate,  LADDERS * intersections);

    // plane 23
    if (blacks_move) {
        std::fill(iterate, iterate+intersections, static_cast<float>(true));
    }
    std::advance(iterate,  intersections);

    // plane 24
    std::fill(iterate, iterate+intersections, static_cast<float>(true));
    std::advance(iterate,  intersections);

    assert(iterate == std::end(input_data));

    return input_data;
}

std::vector<float> Model::gather_features(const GameState *const state) {

    static constexpr auto FEATURE_PASS = 6;
    static constexpr auto FEATURE_KO = 4;
    // static constexpr auto FEATURE_MISC = 2;

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

    // const auto komi = state->board.get_komi();
    // input_data[roll] = komi / 15.0f;
  
    // const auto boardsize = state->board.get_boardsize();
    // input_data[roll + 1] = boardsize / 5.0f;
    // roll += FEATURE_MISC;

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

void Model::features_stream(std::ostream &out, const GameState *const state, const int symmetry) {

    const auto planes = gather_planes(state, symmetry);
    const auto features = gather_features(state);
    const auto boardsize = state->board.get_boardsize();
    const auto intersections = boardsize * boardsize;
    const auto input_channels = INPUT_CHANNELS;
    const auto input_features = INPUT_FEATURES;

    for (auto i = size_t{0}; i < input_channels; ++i) {
        out << std::endl;
        out << "patten : " << (i+1) << std::endl;
        for (int idx = 0; idx < intersections; ++idx) {
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

std::string Model::features_to_string(GameState &state, const int symmetry) {

    auto out = std::ostringstream{};
    features_stream(out, &state, symmetry);

    return out.str();
}

NNResult Model::get_result(const GameState *const state,
                           std::vector<float> &policy,
                           std::vector<float> &score_belief,
                           std::vector<float> &ownership,
                           std::vector<float> &final_score,
                           std::vector<float> &values,
                           const float softmax_temp,
                           const int symmetry) {
    NNResult result;

    const auto intersections = state->get_intersections();

    // Probabilities
    const auto probabilities = Activation::Softmax(policy, softmax_temp);
    for (int idx = 0; idx < intersections; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        result.policy[sym_idx] = probabilities[idx];
    }
    result.policy_pass = probabilities[intersections];

    // Score belief
    (void) score_belief;
  
    // Ownership
    for (int idx = 0; idx < intersections; ++idx) {
        const auto sym_idx = Board::symmetry_nn_idx_table[symmetry][idx];
        result.ownership[sym_idx] = std::tanh(ownership[idx]);
    }

    // Final score
    result.final_score = 20 * final_score[0];

    // Winrate misc
    result.alpha = values[0];
    result.beta = values[1];
    // result.gamma = values[2];

    return result;
}

float Model::get_winrate(GameState &state, const NNResult &result) {
    const auto komi = state.get_komi();
    const auto color = state.get_to_move();
    const auto current_komi = (color == Board::BLACK ? komi : -komi);
    return get_winrate(state, result, current_komi);
}

float Model::get_winrate(GameState &state, const NNResult &result, float current_komi) {
    const auto intersections = state.get_intersections();
    const auto alpha = result.alpha;
    const auto beta = std::exp(result.beta) * 10.f / intersections;
    const auto gamme = result.gamma;
    auto winrate = std::tanh((beta * (alpha - current_komi)) + gamme);
    return (winrate + 1.0f) / 2.0f;
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
