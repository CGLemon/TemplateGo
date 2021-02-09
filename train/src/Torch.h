#ifndef TORCH_H_INCLUDE
#define TORCH_H_INCLUDE

#ifdef USE_TORCH
#include <torch/torch.h>

#include <array>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace Torch {

/*
 *  ===================================================
 *  The utils for torch tensor
 */


/*
 * Get the shape of torch tensor.
 */
std::vector<size_t> get_tensor_shape(torch::Tensor &tensor);

/*
 * Get the number of torch tensor elements.
 */
size_t get_tensor_size(torch::Tensor &tensor);

/*
 * Transform the torch tensor to stream.
 */
void tensor_stream(std::ostream &out, torch::Tensor &tensor);

/*
 * Transform the torch tensor to C++ std::vector.
 */
std::vector<float> tensor_to_vector(torch::Tensor &tensor);

/*
 * ===================================================
 */



/*
 * Save network weights to the file.
 */
void save_weights(std::string &filename, std::vector<std::vector<float>> &weights);

/*
 * Get network weights form the file.
 */
void load_weights(std::string &filename, std::vector<std::vector<float>> &weights);



static constexpr int CONV_WEIGHT = 1;
static constexpr int BATCHNORM_VAR = 2;
static constexpr int BATCHNORM_MEAN = 3;
static constexpr int FULLYCONNET_WEIGHT = 4;
static constexpr int FULLYCONNET_BIAS = 5;

struct Net : torch::nn::Module {

    Net(size_t boardsize,
        size_t resnet_channels,
        size_t resnet_blocks);

    void load_weights(std::vector<std::vector<float>> &weights);
    std::vector<std::vector<float>> gather_weights();

    size_t get_input_channels();
    size_t get_input_features();
    size_t get_value_misc();
  
    size_t m_boardsize{0};
    size_t m_input_channels{0};
    size_t m_res_channels{0};
    size_t m_res_blocks{0};
    size_t m_squeeze_size{0};
  
    size_t m_input_features{0};
    size_t m_policy_out{0};
    size_t m_value_out{0};

    size_t m_probbility_out{0};
    size_t m_scorebelief_out{0};
    size_t m_ownership_out{0};
    size_t m_value_misc{0};
    size_t m_finalscore_misc{0};

    float m_eps;
    bool m_train_mode;
    bool m_affine;

    bool applied;
    void dump_stack();

    void build_input_layer();
    void build_tower_block(size_t id);
    void build_tower();
    void build_policy_head();
    void build_value_head();

    void build_graph();

    std::vector<std::pair<torch::Tensor, int>> m_collect;
 
    torch::Tensor input_forward(torch::Tensor planes, torch::Tensor features);
    torch::Tensor tower_forward(torch::Tensor x, size_t id);
    torch::Tensor se_forward(torch::Tensor x, torch::Tensor res, size_t id);
    std::vector<torch::Tensor> value_forward(torch::Tensor x);
    std::vector<torch::Tensor> policy_forward(torch::Tensor x);

    std::array<torch::Tensor, 6> forward(torch::Tensor planes, torch::Tensor features);

    struct RES_BLOCK {
        torch::nn::Conv2d Convlayer_1{nullptr};
        torch::nn::BatchNorm2d BNlyer_1{nullptr};
        torch::nn::Conv2d Convlayer_2{nullptr};
        torch::nn::BatchNorm2d BNlyer_2{nullptr};

        torch::nn::Linear ExtendFCLayer{nullptr};
        torch::nn::Linear SqueezeFCLayer{nullptr};
    };

    // input layer
    torch::nn::Conv2d InputLayer{nullptr};
    torch::nn::BatchNorm2d InputBNlyer{nullptr};

    torch::nn::Linear InputFCLayer{nullptr};

    // tower
    std::vector<RES_BLOCK> ResidualTower;

    // policy head
    torch::nn::Conv2d PolicyHead{nullptr};
    torch::nn::BatchNorm2d PolicyBNlyer{nullptr};

    torch::nn::Conv2d OppPolicyConv{nullptr};    // opponent's probabilities without pass
    torch::nn::Linear OppPolicyFCLayer{nullptr}; // opponent's pass 

    torch::nn::Conv2d PolicyConv{nullptr};    // probabilities without pass
    torch::nn::Linear PolicyFCLayer{nullptr}; // pass 

    // value head 
    torch::nn::Conv2d ValueHead{nullptr};
    torch::nn::BatchNorm2d ValueBNlyer{nullptr};

    torch::nn::Conv2d ScoreBeliefConv{nullptr};   // score belief
    torch::nn::Conv2d OwnershipConv{nullptr};     // ownship
    torch::nn::Linear FinalScoreFCLayer{nullptr}; // final score
    torch::nn::Linear ValueFCLayer{nullptr};      // winrate
};


struct TrainDataBuffer {
    std::vector<float> input_planes;
    std::vector<float> input_features;

    std::vector<float> probabilities;
    std::vector<float> opponent_probabilities;
    int scorebelief_idx;
    float final_score;
    float current_komi;

    std::vector<float> ownership;
    std::vector<float> winrate;
};

struct TrainConfig {
    size_t in_channels;
    size_t in_features;
    size_t resnet_channels;
    size_t resnet_blocks;
    size_t boardsize;
    float learning_rate;
    float weight_decay;

    bool force_cpu;
};


class Train_helper {
public:
    Train_helper() = default;
    ~Train_helper();

    void init(TrainConfig config);

    std::vector<std::pair<std::string, float>>
    train_batch(std::vector<TrainDataBuffer> &buffer);

    void save_weights(std::string &filename);
    void change_config(TrainConfig config);
private:
    bool gpu_is_available{false};
    bool on_gpu{false};
    std::shared_ptr<Net> m_net{nullptr};

    torch::Device *m_device{nullptr};
    torch::Device *m_host{nullptr};

    torch::optim::Adam *m_optimizer{nullptr};

    TrainConfig m_config;
};

class Evaluation {
public:
    void build(size_t boardsize,
               size_t resnet_channels,
               size_t resnet_blocks);

    void fill_weights(std::vector<std::vector<float>> &weights);

    void dump_eval(torch::Tensor planes, torch::Tensor features, float current_komi);

    std::shared_ptr<Net> m_net{nullptr};
};

}
#endif
#endif
