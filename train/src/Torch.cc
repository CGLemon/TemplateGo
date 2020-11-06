#include "Torch.h"
#include <algorithm>
#include <cassert>
#ifdef USE_TORCH
namespace Torch {

/*
 * Get the shape of torch tensor.
 */
std::vector<size_t> get_tensor_shape(torch::Tensor &tensor) {
    auto res = std::vector<size_t>{};

    const size_t dim_size = tensor.dim();
    for (auto d = size_t{0}; d < dim_size; ++d) {
        res.emplace_back(tensor.size(d));
    }

    return res;
}

/*
 * Get the number of torch tensor elements.
 */
size_t get_tensor_size(torch::Tensor &tensor) {
    auto tensor_shape = get_tensor_shape(tensor);
    size_t sz = 1;
    for (auto &s : tensor_shape) {
        sz *= s;
    }
    return sz;
}

/*
 * Transform the torch tensor to stream.
 */
void tensor_stream(std::ostream &out, torch::Tensor &tensor) {
    size_t size = get_tensor_size(tensor);
    auto tensor_ptr = (float*)tensor.data_ptr();

    out << *tensor_ptr;
    tensor_ptr++;

    for (auto i = size_t{1}; i < size; ++i) {
        out << " " << *tensor_ptr;
        tensor_ptr++;
    }
}

/*
 * Transform the torch tensor to C++ std::vector.
 */
std::vector<float> tensor_to_vector(torch::Tensor &tensor) {
    size_t size = get_tensor_size(tensor);

    auto res = std::vector<float>(size, 0.0f);
    auto tensor_ptr = (float*)tensor.data_ptr();
    for (auto i = size_t{0}; i < size; ++i) {
        res[i] = *tensor_ptr;
        tensor_ptr++;
    }
    return res;
}

/*
 * Save network weights to the file.
 */
void save_weights(std::string &filename, std::vector<std::vector<float>> &weights) {
    std::ofstream file;
    file.open(filename.c_str());
    size_t cnt = 0;
    for (const auto &w : weights) {
        // for (auto &p : w) {
        //    file << p << " ";
        // }
        auto pbegin = std::begin(w);
        auto pend   = std::end(w);
        file << *pbegin;
        pbegin = std::next(pbegin, 1);
        if (pbegin != pend) {
            std::for_each(pbegin, pend, [&](auto p){ file << " " << p; } );
        }

        if (cnt != weights.size()-1) {
            file << std::endl;
        }
        cnt++;
    }
    file.close();
}


/*
 * Get network weights form the file.
 */
void load_weights(std::string &filename, std::vector<std::vector<float>> &weights) {
    if (!weights.empty()) {
         return;
    }

    std::ifstream file;
    std::stringstream buffer;
    std::string line;

    file.open(filename.c_str());
    while(std::getline(file, line)) {
        buffer << line;
        buffer << std::endl;
    }
    file.close();

    while(std::getline(buffer, line)) {
        std::vector<float> weight;
        float para;
        std::stringstream line_buffer(line);
        while(line_buffer >> para) {
            weight.emplace_back(para);
        }
        weights.emplace_back(std::move(weight));
    }
}


Net::Net(size_t boardsize, 
         size_t resnet_channels,
         size_t resnet_blocks) {

    m_input_channels = 24;
    m_input_features = 10;
    m_boardsize = boardsize;

    // Rediual tower
    m_res_channels = resnet_channels;
    m_res_blocks = resnet_blocks;
    m_squeeze_size = 4 * resnet_channels; // se unit

    // Output head
    m_policy_out = 32;
    m_value_out = 64;
    m_probbility_out = 1;
    m_scorebelief_out = 2;
    m_ownership_out = 1;
    m_value_misc = 2;
    m_finalscore_misc = 1;

    // BatchNorm Layer
    m_eps = 1e-05;
    m_train_mode = true;
    m_affine = false;

    applied = false;
}

size_t Net::get_input_channels() {
    return m_input_channels;
}

size_t Net::get_input_features() {
    return m_input_features;
}

size_t Net::get_value_misc() {
    return m_value_misc;
}


/*
 * Building input layer graph.
 */
void Net::build_input_layer() {
    InputLayer = register_module("Input convolution layer",
                                 torch::nn::Conv2d(
                                 torch::nn::Conv2dOptions(
                                  m_input_channels, m_res_channels, 3)
                                 .stride(1)
                                 .padding(1)
                                 .bias(false)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{InputLayer->weight, CONV_WEIGHT});


    InputBNlyer = register_module("Input batchNorm layer",
                                  torch::nn::BatchNorm2d(
                                  torch::nn::BatchNorm2dOptions(m_res_channels)
                                  .eps(m_eps)
                                  .affine(m_affine)
                                  .track_running_stats(m_train_mode)));

    m_collect.emplace_back(std::pair<torch::Tensor, int>{InputBNlyer->running_mean, BATCHNORM_MEAN});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{InputBNlyer->running_var, BATCHNORM_VAR});
  
    InputFCLayer = register_module("Input fullyconnect layer",
                                   torch::nn::Linear(m_input_features, m_res_channels));

    m_collect.emplace_back(std::pair<torch::Tensor, int>{InputFCLayer->weight, FULLYCONNET_WEIGHT});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{InputFCLayer->bias, FULLYCONNET_BIAS});
}


/*
 * Building rediual tower block graph.
 */
void Net::build_tower_block(size_t id) {

   /*
    * TODO:
    * Fixup Initialization: Residual Learning Withuot Normalization
    * We can simply remove BatchNorm layer because seemed to have no disadvatanges in the learning.
    * Most importantly, BatchNorm layer is expensive to compute. Removing them can speed up learning.
    *
    * The SE Unit seemed to have same effect with BatchNorm, maybe?
    */

    auto tower_ptr = ResidualTower.data() + id;
    assert(ResidualTower.size() > id);

    std::string name_1 = "tower convolution layer 1 : " + std::to_string(id);
    tower_ptr->Convlayer_1 = register_module(name_1,
                                             torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(
                                             m_res_channels, m_res_channels, 3)
                                             .stride(1)
                                             .padding(1)
                                             .bias(false)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->Convlayer_1->weight, CONV_WEIGHT});


    std::string name_2 = "tower batchNorm layer 1 : " + std::to_string(id);
    tower_ptr->BNlyer_1 = register_module(name_2,
                                          torch::nn::BatchNorm2d(
                                          torch::nn::BatchNorm2dOptions(m_res_channels)
                                          .eps(m_eps)
                                          .affine(m_affine)
                                          .track_running_stats(m_train_mode)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->BNlyer_1->running_mean, BATCHNORM_MEAN});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->BNlyer_1->running_var, BATCHNORM_VAR});
 

    std::string name_3 = "tower convolution layer 2 : " + std::to_string(id);
    tower_ptr->Convlayer_2 = register_module(name_3,
                                             torch::nn::Conv2d(
                                             torch::nn::Conv2dOptions(
                                             m_res_channels, m_res_channels, 3)
                                             .stride(1)
                                             .padding(1)
                                             .bias(false)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->Convlayer_2->weight, CONV_WEIGHT});


    std::string name_4 = "tower batchNorm layer 2 : " + std::to_string(id);
    tower_ptr->BNlyer_2 = register_module(name_4,
                                          torch::nn::BatchNorm2d(
                                          torch::nn::BatchNorm2dOptions(m_res_channels)
                                          .eps(m_eps)
                                          .affine(m_affine)
                                          .track_running_stats(m_train_mode)));

    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->BNlyer_2->running_mean, BATCHNORM_MEAN});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->BNlyer_2->running_var, BATCHNORM_VAR});
  

    std::string name_5 = "tower extend fullyconnect layer : " + std::to_string(id);
    tower_ptr->ExtendFCLayer = register_module(name_5,
                                               torch::nn::Linear(m_res_channels, m_squeeze_size));

    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->ExtendFCLayer->weight, FULLYCONNET_WEIGHT});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->ExtendFCLayer->bias, FULLYCONNET_BIAS});


    std::string name_6 = "tower squeeze fullyconnect layer 1 : " + std::to_string(id);
    tower_ptr->SqueezeFCLayer = register_module(name_6,
                                                torch::nn::Linear(m_squeeze_size, 2 * m_res_channels));

    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->SqueezeFCLayer->weight, FULLYCONNET_WEIGHT});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{tower_ptr->SqueezeFCLayer->bias, FULLYCONNET_BIAS});
}

/*
 * Building rediual tower graph.
 */
void Net::build_tower() {
    for (auto b = size_t{0}; b < m_res_blocks; ++b) {
        ResidualTower.emplace_back(RES_BLOCK{});
    }
    for (auto b = size_t{0}; b < m_res_blocks; ++b) {
        build_tower_block(b);
    } 
}

/*
 * Building policy head graph.
 */
void Net::build_policy_head() {

    PolicyHead = register_module("policy head convolution",
                                 torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                 m_res_channels, m_policy_out, 1)
                                 .stride(1)
                                 .bias(false)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{PolicyHead->weight, CONV_WEIGHT});

    PolicyBNlyer = register_module("policy head batchNorm",
                                   torch::nn::BatchNorm2d(
                                   torch::nn::BatchNorm2dOptions(m_policy_out)
                                   .eps(m_eps)
                                   .affine(m_affine)
                                   .track_running_stats(m_train_mode)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{PolicyBNlyer->running_mean, BATCHNORM_MEAN});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{PolicyBNlyer->running_var, BATCHNORM_VAR});

  
    PolicyConv = register_module("policy convolution",
                                 torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                 m_policy_out, m_probbility_out, 1)
                                 .stride(1)
                                 .bias(false)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{PolicyConv->weight, CONV_WEIGHT});


    PolicyFCLayer = register_module("policy fully connect",
                                   torch::nn::Linear(m_policy_out, 1));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{PolicyFCLayer->weight, FULLYCONNET_WEIGHT});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{PolicyFCLayer->bias, FULLYCONNET_BIAS});


   /*
    * "OppPolicyConv" and "OppPolicyFCLayer" predict opponent's policy.
    * They only exist in training but never use in Template Go forward pipe.
    * So, we don't need to collect the weights.
    */
    OppPolicyConv = register_module("opponent policy convolution",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                    m_policy_out, m_probbility_out, 1)
                                    .stride(1)
                                    .bias(false)));

    OppPolicyFCLayer = register_module("opponent policy fully connect",
                                       torch::nn::Linear(m_policy_out, 1));

}

/*
 * Building value head graph.
 */
void Net::build_value_head() {
    ValueHead = register_module("value head convolution",
                                 torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                 m_res_channels, m_value_out, 1)
                                 .stride(1)
                                 .bias(false)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{ValueHead->weight, CONV_WEIGHT});

    ValueBNlyer = register_module("value batchNorm",
                                   torch::nn::BatchNorm2d(
                                   torch::nn::BatchNorm2dOptions(m_value_out)
                                   .eps(m_eps)
                                   .affine(m_affine)
                                   .track_running_stats(m_train_mode)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{ValueBNlyer->running_mean, BATCHNORM_MEAN});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{ValueBNlyer->running_var, BATCHNORM_VAR});
  

    ScoreBeliefConv = register_module("final score convolution",
                                      torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                      m_value_out, m_scorebelief_out, 1)
                                      .stride(1)
                                      .bias(false)));
     m_collect.emplace_back(std::pair<torch::Tensor, int>{ScoreBeliefConv->weight, CONV_WEIGHT});

    OwnershipConv = register_module("ownership convolution",
                                    torch::nn::Conv2d(torch::nn::Conv2dOptions(
                                    m_value_out, m_ownership_out, 1)
                                    .stride(1)
                                    .bias(false)));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{OwnershipConv->weight, CONV_WEIGHT});

    FinalScoreFCLayer = register_module("final score connect",
                                         torch::nn::Linear(m_value_out, m_finalscore_misc));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{FinalScoreFCLayer->weight, FULLYCONNET_WEIGHT});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{FinalScoreFCLayer->bias, FULLYCONNET_BIAS});

    ValueFCLayer = register_module("value fully connect",
                                 torch::nn::Linear(m_value_out, m_value_misc));
    m_collect.emplace_back(std::pair<torch::Tensor, int>{ValueFCLayer->weight, FULLYCONNET_WEIGHT});
    m_collect.emplace_back(std::pair<torch::Tensor, int>{ValueFCLayer->bias, FULLYCONNET_BIAS});
}

/*
 * Building graph.
 */
void Net::build_graph() {
    if (applied) {
        return;
    }
    applied = true;
    build_input_layer();
    build_tower();
    build_policy_head();
    build_value_head();
}

torch::Tensor Net::input_forward(torch::Tensor planes, torch::Tensor features) {
    auto x = InputLayer(planes); 
    x = InputBNlyer(x);

    auto b = InputFCLayer(features);
    b = torch::reshape(b, {b.size(0), b.size(1), 1, 1});

    return torch::relu(x + b);
}


torch::Tensor Net::se_forward(torch::Tensor x, torch::Tensor res, size_t id) {
    auto tower_ptr = ResidualTower.data() + id;
    assert(ResidualTower.size() > id);

    auto out = torch::adaptive_avg_pool2d(x, {1, 1});
    out = torch::flatten(out, 1, 3);
    out = tower_ptr->ExtendFCLayer(out);
    out = torch::relu(out);
    out = tower_ptr->SqueezeFCLayer(out);

    auto split_tensors = torch::split(out, m_res_channels, 1);
  
    auto gamma = torch::sigmoid(split_tensors[0]);
    auto beta = split_tensors[1];

    gamma = torch::reshape(gamma, {gamma.size(0), gamma.size(1), 1, 1});
    beta = torch::reshape(beta, {beta.size(0), beta.size(1), 1, 1});

    return gamma * x + beta + res;
}

torch::Tensor Net::tower_forward(torch::Tensor x, size_t id) {
 
   /*
    * TODO: Removing BatchNorm Layer can speed up the learning.
    */
    auto tower_ptr = ResidualTower.data() + id;
    assert(ResidualTower.size() > id);

    auto out = tower_ptr->Convlayer_1(x);
    out = tower_ptr->BNlyer_1(out);
    out = torch::relu(out);

    out = tower_ptr->Convlayer_2(out);
    out = tower_ptr->BNlyer_2(out);

    out = se_forward(out, x, id);
    out = torch::relu(out);

    return out;
}

std::vector<torch::Tensor> Net::policy_forward(torch::Tensor x) {
    x = PolicyHead(x);
    x = PolicyBNlyer(x);
    x = torch::relu(x);

    auto p_pool = torch::adaptive_avg_pool2d(x, {1, 1});
    auto pass = torch::flatten(p_pool, 1, 3);
    pass = PolicyFCLayer(pass);

    auto prob = PolicyConv(x);
    prob = torch::flatten(prob, 1, 3);
    prob =  torch::cat({prob, pass}, 1);

    auto opp_pass = torch::flatten(p_pool, 1, 3);
    opp_pass = OppPolicyFCLayer(opp_pass); 

    auto opp_prob = OppPolicyConv(x);
    opp_prob = torch::flatten(opp_prob, 1, 3);
    opp_prob =  torch::cat({opp_prob, opp_pass}, 1);

    return std::vector<torch::Tensor>{prob, opp_prob};
}

std::vector<torch::Tensor> Net::value_forward(torch::Tensor x) {
    x = ValueHead(x);
    x = ValueBNlyer(x);
    x = torch::relu(x);

    auto scorebelief = ScoreBeliefConv(x);
    scorebelief = torch::flatten(scorebelief, 1, 3);

    auto ownership = OwnershipConv(x);
    ownership = torch::flatten(ownership, 1, 3);
    ownership = torch::tanh(ownership);

    auto v_pool = torch::adaptive_avg_pool2d(x, {1, 1});
    auto finalscore = torch::flatten(v_pool, 1, 3);
    finalscore = FinalScoreFCLayer(finalscore);

    auto winrate_misc = torch::flatten(v_pool, 1, 3);
    winrate_misc = ValueFCLayer(winrate_misc);

    return std::vector<torch::Tensor>{scorebelief, ownership, finalscore, winrate_misc};
}

std::array<torch::Tensor, 6> Net::forward(torch::Tensor planes, torch::Tensor features) {
    auto x = input_forward(planes, features); 
    for (auto b = size_t{0}; b < m_res_blocks; ++b) {
        x = tower_forward(x, b);
    }
    // policy head
    auto pol = policy_forward(x);

    // one shot, the probabilities of all possible move, include pass.
    auto prob = pol[0];

    // one shot, the probabilities of all opponent possible move, include pass.
    auto opp_prob = pol[1];


    // value head
    auto val = value_forward(x);

   /*
    * Score belief predicts the board on score without komi.
    * If the board size is "bsize", the tatol predicts score is "bsize * bsize * 2".
    * Set "s = bsize * bsize" for each possible final score value
    *      [-s, -(s-1), -(s-2), ...., s-2, s-1] 
    */
    auto scorebelief = val[0]; 

   /*
    * Ownership of intersections.
    * If value is 1, that means the point is belong to current player.
    * If value is -1, that means the point is belong to opponent player.
    * If value is 0, that means the point is seki.
    */
    auto ownership = val[1];  

    // Final score predicts the board on score without komi.
    auto finalscore = val[2];

    // Winrate Misc
    auto winrate_misc = val[3];  

    return std::array<torch::Tensor, 6>{prob, opp_prob, scorebelief, ownership, finalscore, winrate_misc};
}

std::vector<std::vector<float>> Net::gather_weights() {

    auto out = std::vector<std::vector<float>>{};
    for (auto &p : m_collect) {
        out.emplace_back(std::move(tensor_to_vector(p.first)));
    }
    return out;
}

void Net::load_weights(std::vector<std::vector<float>> &loadweights) {
    auto id = size_t{0};

    for (auto & collect : m_collect) {
        auto weights = collect.first;
        auto tensor_ptr = (float*)weights.data_ptr();
        auto size = get_tensor_size(weights);
        assert(size == loadweights[id].size());

        auto loadweights_ptr = loadweights[id].data();
        for (auto i = size_t{0}; i < size; ++i) {
            *tensor_ptr = *loadweights_ptr;
            tensor_ptr++;
            loadweights_ptr++;
        }
        id++;
    }

    assert(loadweights.size() == id);
}

void Net::dump_stack() {
    const auto type2string = [=](int type) -> std::string {
        if (type == CONV_WEIGHT) {
            return std::string{"Convolution weights"};
        } else if (type == BATCHNORM_VAR) {
            return std::string{"BatchNorm var"};
        } else if (type == BATCHNORM_MEAN) {
            return std::string{"BatchNorm mean"};
        } else if (type == FULLYCONNET_WEIGHT) {
            return std::string{"FulliyConnect weights"};
        } else if (type == FULLYCONNET_BIAS) {
            return std::string{"FulliyConnect biases"};
        } else {
            return std::string{"Not defined"};
        }
    };

    const auto shape2string = [](std::vector<size_t> &tensor_shape) -> std::string {
        auto res = std::string{"shape "};
        for (const auto &s : tensor_shape) {
            res += std::to_string(s) + " ";
        }
        return res;
    };


    for (auto &p : m_collect) {
        auto shape = get_tensor_shape(p.first);
        auto type = p.second;

        printf("%s\n", shape2string(shape).c_str());
        printf("%s\n", type2string(type).c_str());
        printf("\n");
    }
}


Train_helper::~Train_helper() {
    if (m_optimizer != nullptr) {
        delete m_optimizer;
    }
    if (m_host != nullptr) {
        delete m_host;
    }
    if (m_device != nullptr) {
        delete m_device;
    }
}

void Train_helper::init(TrainConfig config) {

    if (m_host != nullptr) {
        return;
    } 

    m_config = config;
    const size_t in_channels = m_config.in_channels;
    const size_t in_features = m_config.in_features;
    const size_t resnet_channels = m_config.resnet_channels;
    const size_t resnet_blocks = m_config.resnet_blocks;
    const size_t boardsize = m_config.boardsize;
    const bool force_cpu = m_config.force_cpu;

    if (m_net == nullptr) {
        m_net = std::make_shared<Net>(boardsize, resnet_channels, resnet_blocks);
    }
  
    // Be sure that planes channels be equal to the netork input channels.
    assert(m_net->get_input_channels() == in_channels);

    // Be sure that input features be equal to the netork input features.
    assert(m_net->get_input_features() == in_features);


    // First. Build network.
    m_net->build_graph();

    // Second. Set learning rate and weight decay.
    float lr = m_config.learning_rate;
    float weight_decay = m_config.weight_decay;
    if (m_optimizer == nullptr) {
        m_optimizer = new torch::optim::Adam(m_net->parameters(),
                              torch::optim::AdamOptions(lr).weight_decay(weight_decay));
    }


    // Third. Check the GPU is available or not.
    m_host = new torch::Device(torch::kCPU);
    on_gpu = false;

    if (torch::cuda::is_available() && !force_cpu) {
        gpu_is_available = true;
        m_device = new torch::Device(torch::kCUDA);
    } else {
        gpu_is_available = false;
        m_device = new torch::Device(torch::kCPU);
    }

    if (gpu_is_available && !on_gpu) {
        // Push the network form CPU to GPU.
        m_net->to(*m_device);
        on_gpu = true;
    }
}

void Train_helper::save_weights(std::string &filename) {
    if (gpu_is_available && on_gpu) {
        // Becuase the computer can not get weights form GPU,
        // we need to push the network form GPU to CPU first.
        on_gpu = false;
        m_net->to(*m_host);
    }

    auto weights = m_net->gather_weights();
    Torch::save_weights(filename, weights);
}

void Train_helper::change_config(TrainConfig config) {

    float old_lr = m_config.learning_rate;
    float old_weight_decay = m_config.weight_decay;

    m_config = config;
    float lr = m_config.learning_rate;
    float weight_decay = m_config.weight_decay;
    if (m_optimizer == nullptr) {
        m_optimizer = new torch::optim::Adam(m_net->parameters(),
                              torch::optim::AdamOptions(lr).weight_decay(weight_decay));
    } else if (old_lr != lr || old_weight_decay != weight_decay) {
        m_optimizer->zero_grad();
        delete m_optimizer;
        m_optimizer = new torch::optim::Adam(m_net->parameters(),
                              torch::optim::AdamOptions(lr).weight_decay(weight_decay));
    }
}

std::vector<std::pair<std::string, float>>
Train_helper::train_batch(std::vector<TrainDataBuffer> &buffer) {
    if (on_gpu == false && gpu_is_available == true) {
        m_net->to(*m_device);
        on_gpu = true;
    }
    const int batch_size = buffer.size();
    const int boardsize = m_config.boardsize;
    const int in_channels = m_config.in_channels;
    const int in_features = m_config.in_features;
    const int intersections = boardsize * boardsize;

    if (batch_size == 0) {
        return std::vector<std::pair<std::string, float>> {{"_NO_LOSS_", 0}};
    }

    auto buffer_ptr = std::begin(buffer);

    // input
    const int intput_planes_size = buffer_ptr->input_planes.size();
    const int intput_features_size = buffer_ptr->input_features.size();
  
    // target
    const int probabilities_size = buffer_ptr->probabilities.size();
    const int scorebelief_size = 2 * intersections;
    const int finalscore_size = 1;
    const int ownership_size = buffer_ptr->ownership.size();
    const int winrate_size = buffer_ptr->winrate.size();
  

    // Be sure size of datas are correct. 
    assert(intput_planes_size == in_channels * intersections);
    assert(intput_features_size == in_features);
    assert(probabilities_size ==  1 + intersections);
    assert(ownership_size ==  intersections);
    assert(winrate_size == 1);

    // input
    torch::Tensor batch_input_planes = torch::zeros({batch_size, in_channels, boardsize, boardsize});
    torch::Tensor batch_input_features = torch::zeros({batch_size, in_features});

    auto batch_input_planes_ptr = (float*)batch_input_planes.data_ptr();
    auto batch_input_featurest_ptr = (float*)batch_input_features.data_ptr();

    // target
    torch::Tensor batch_policy = torch::zeros({batch_size, probabilities_size});
    torch::Tensor batch_opp_policy = torch::zeros({batch_size, probabilities_size});
    torch::Tensor batch_scorebelief = torch::zeros({batch_size, scorebelief_size});
    torch::Tensor batch_finalscore = torch::zeros({batch_size, finalscore_size});
    torch::Tensor batch_ownership = torch::zeros({batch_size, ownership_size});
    torch::Tensor current_komi = torch::zeros({batch_size, 1});
    torch::Tensor batch_winrate = torch::zeros({batch_size, winrate_size});
  
    auto batch_policy_ptr = (float*)batch_policy.data_ptr();
    auto batch_opp_policy_ptr = (float*)batch_opp_policy.data_ptr();
    auto batch_scorebelief_ptr = (float*)batch_scorebelief.data_ptr();
    auto batch_finalscore_ptr = (float*)batch_finalscore.data_ptr();
    auto batch_ownership_ptr = (float*)batch_ownership.data_ptr();
    auto current_komi_ptr = (float*)current_komi.data_ptr();
    auto batch_winrate_ptr = (float*)batch_winrate.data_ptr();
  
    for (auto b = int{0}; b < batch_size; ++b) {
        // input
        for (auto idx = int{0}; idx < intput_planes_size; ++idx) {
            *(batch_input_planes_ptr+idx) = buffer_ptr->input_planes[idx];
        }
        for (auto idx = int{0}; idx < intput_features_size; ++idx) {
            *(batch_input_featurest_ptr+idx) = buffer_ptr->input_features[idx];
        }

        // target
        for (auto idx = int{0}; idx < probabilities_size; ++idx) {
            *(batch_policy_ptr+idx) = buffer_ptr->probabilities[idx];
        }

        for (auto idx = int{0}; idx < probabilities_size; ++idx) {
            *(batch_opp_policy_ptr+idx) = buffer_ptr->opponent_probabilities[idx];
        }

        *(batch_scorebelief_ptr + (buffer_ptr->scorebelief_idx)) = 1.0f;

        *batch_finalscore_ptr = buffer_ptr->final_score;

        for (auto idx = int{0}; idx < ownership_size; ++idx) {
            *(batch_ownership_ptr+idx) = buffer_ptr->ownership[idx];
        }

        *current_komi_ptr = buffer_ptr->current_komi;

        for (auto idx = int{0}; idx < winrate_size; ++idx) {
            *(batch_winrate_ptr+idx) = buffer_ptr->winrate[idx];
        }

        buffer_ptr++;
        // input
        batch_input_planes_ptr += intput_planes_size;
        batch_input_featurest_ptr += intput_features_size;

        // target
        batch_policy_ptr += probabilities_size;
        batch_opp_policy_ptr += probabilities_size;
        batch_scorebelief_ptr += scorebelief_size;
        batch_finalscore_ptr += finalscore_size;
        batch_ownership_ptr += ownership_size;
        current_komi_ptr += 1;
        batch_winrate_ptr += winrate_size;
    }
  

    if (gpu_is_available && on_gpu) {
        batch_input_planes = batch_input_planes.to(*m_device);
        batch_input_features = batch_input_features.to(*m_device);

        batch_policy = batch_policy.to(*m_device);
        batch_opp_policy = batch_opp_policy.to(*m_device);
        batch_scorebelief = batch_scorebelief.to(*m_device);
        batch_finalscore = batch_finalscore.to(*m_device);
        batch_ownership = batch_ownership.to(*m_device);
        current_komi = current_komi.to(*m_device);
        batch_winrate = batch_winrate.to(*m_device);
    }

    m_optimizer->zero_grad();
    auto prediction = m_net->forward(batch_input_planes, batch_input_features);
    auto policy = prediction[0];
    auto opp_policy = prediction[1];
    auto scorebelief = prediction[2];
    auto ownership = prediction[3];
    auto finalscore = prediction[4];
    auto winrate_misc = prediction[5];

    auto policy_loss = torch::mean(-torch::sum(
                           torch::mul(torch::log_softmax(policy, -1), batch_policy), // unreduce
                               1), 0);

    auto opp_policy_loss = torch::mean(-torch::sum(
                               torch::mul(torch::log_softmax(opp_policy, -1), batch_opp_policy), // unreduce
                                   1), 0);

    auto scorebelief_cdf_loss = torch::mse_loss(
                                    torch::cumsum(torch::softmax(scorebelief, -1), 1), torch::cumsum(batch_scorebelief, 1));

    auto scorebelief_pdf_loss = torch::mean(-torch::sum(
                                    torch::mul(torch::log_softmax(scorebelief, -1), batch_scorebelief), // unreduce
                                        1), 0);

    const auto huber_loss = [](torch::Tensor &x, torch::Tensor &y, float delta){
        auto absdiff = torch::abs(x - y);
        auto res = torch::where(absdiff > delta,
                                    0.5f * delta * delta + delta * (absdiff - delta),
                                    0.5f * torch::mul(absdiff, absdiff));
        return torch::mean(torch::mean(res, 1), 0);
    };

    const auto misc2winrate = [](torch::Tensor misc, torch::Tensor c_komi, float intersections) {
        auto miscs = torch::chunk(misc, 2, 1);
        // miscs[0] : alpha
        // miscs[1] : beta
        // miscs[2] : gamma
        return torch::tanh((torch::exp(miscs[1]) * 10.f / intersections) * (miscs[0] - c_komi) /* + miscs[2] */);
    };

    const auto MSE = [](torch::Tensor x, float target) {
        auto e = x - target;
        auto res = torch::mul(e, e);
        return torch::mean(torch::mean(res, 1), 0);
    };


    auto finalscore_mean = 20 * finalscore;
    auto finalscore_loss = huber_loss(finalscore_mean, batch_finalscore, 12.f);

    auto ownership_loss = torch::mean(-torch::sum(
                              torch::mul(torch::log((ownership + 1)/2 + 0.0001f), batch_ownership), 1), 0) +
                              torch::mean(-torch::sum(torch::mul(torch::log((1 - ownership)/2 + 0.0001f), 1 - batch_ownership), // unreduce
                                  1), 0);

    auto winrate_loss = torch::mse_loss(misc2winrate(winrate_misc, current_komi, (float)intersections), batch_winrate);

    auto loss = 
        1.00f * policy_loss +
        0.15f * opp_policy_loss +
        (1.5f / (float)intersections) * scorebelief_cdf_loss +
        (1.5f / (float)intersections) * scorebelief_pdf_loss +
        0.0012 * finalscore_loss +
        (0.15f / (float)intersections) * ownership_loss +
        1.00f * winrate_loss;

    loss.backward();
    m_optimizer->step();

    std::vector<std::pair<std::string, float>> Loss_;

    Loss_.emplace_back(std::pair<std::string, float>{"total", loss.item<float>()});
    Loss_.emplace_back(std::pair<std::string, float>{"policy | coefficient : 1", policy_loss.item<float>()});
    Loss_.emplace_back(std::pair<std::string, float>{"opponent policy | coefficient : 0.15", opp_policy_loss.item<float>()});
    Loss_.emplace_back(std::pair<std::string, float>{"score belief cdf | coefficient : 1.5/intersections", scorebelief_cdf_loss.item<float>()});
    Loss_.emplace_back(std::pair<std::string, float>{"score belief pdf | coefficient : 1.5/intersections", scorebelief_pdf_loss.item<float>()});
    Loss_.emplace_back(std::pair<std::string, float>{"final score | coefficient : 0.0012", finalscore_loss.item<float>()});
    Loss_.emplace_back(std::pair<std::string, float>{"ownership | coefficient : 0.15/intersections", ownership_loss.item<float>()});
    Loss_.emplace_back(std::pair<std::string, float>{"winrate | coefficient : 1.0", winrate_loss.item<float>()});

    return Loss_;
}

void Evaluation::build(size_t boardsize,
                       size_t resnet_channels,
                       size_t resnet_blocks) {

    if (m_net == nullptr) {
        m_net = std::make_shared<Net>(boardsize, resnet_channels, resnet_blocks);
    }
    m_net->build_graph();
    m_net->eval();
}

void Evaluation::fill_weights(std::vector<std::vector<float>> &weights) {
    m_net->load_weights(weights);
}

void Evaluation::dump_eval(torch::Tensor planes, torch::Tensor features, float current_komi) {
    auto out = m_net->forward(planes, features);

    auto policy = torch::softmax(out[0], -1);
    auto opp_policy = torch::softmax(out[1], -1); // ignore
    auto scorebelief = torch::softmax(out[2], -1);
    auto ownership = out[3];
    auto finalscore = out[4];
    auto winrate_misc = out[5];

    auto policy_vec = tensor_to_vector(policy);
    auto scorebelief_vec = tensor_to_vector(scorebelief);
    auto finalscore_vec = tensor_to_vector(finalscore);
    auto ownership_vec = tensor_to_vector(ownership);
    auto winrate_misc_vec = tensor_to_vector(winrate_misc);

    printf("Policy | size %zu : \n", policy_vec.size());
    for (const auto &p : policy_vec) {
        printf("%.5f ",p);
    }
    printf("\n\n");


    printf("Score Belief | size %zu : \n", scorebelief_vec.size());
    for (const auto &v : scorebelief_vec) {
        printf("%.5f ",v);
    }
    printf("\n\n");

    printf("Final score | size %zu : \n", finalscore_vec.size());
    for (const auto &v : finalscore_vec) {
        printf("%.5f ",v);
    }
    printf("\n\n");


    printf("Ownership | size %zu : \n", ownership_vec.size());
    for (const auto &v : ownership_vec) {
        printf("%.5f ",v);
    }
    printf("\n\n");


    printf("Winrate | size %zu : \n", winrate_misc_vec.size());
    const auto alpha = winrate_misc_vec[0];
    const auto beta = winrate_misc_vec[1];
    const auto gamma = 0.0f; // winrate_misc_vec[2];
    const auto w = std::tanh(((alpha - current_komi)/beta) + gamma);
    printf("alpha : %.5f, beta : %.5f, gamma : %.5f\n", alpha, beta, gamma);
    printf("%.5f",w);
    printf("\n\n");
}

} // namespace Torch 
#endif
