#include "CPUBackend.h"
#include "Utils.h"
template<int BSIZE>
class FORWARD_PIPE {
public:
void forward(std::shared_ptr<Model::NNweights> m_weights,
             const  std::vector<float> &planes,
             const  std::vector<float> &features,
             std::vector<float> &output_pol,
             std::vector<float> &output_sb,
             std::vector<float> &output_os,
             std::vector<float> &output_fs,
             std::vector<float> &output_val) {

    using batchnorm = Batchnorm<BSIZE>;
    using convolve_3 = winograd_convolve3<BSIZE>;
    using convolve_1 = Convolve1<BSIZE>;
    using se_unit = SEUnit<BSIZE>;
    using inputpool = InputPool<BSIZE>;
    using globalpool = GlobalAvgPool<BSIZE>;

    const size_t intersections = BSIZE * BSIZE; 

    size_t output_channels = m_weights->channels;
    size_t input_channels = std::max(static_cast<size_t>(output_channels),
                                     static_cast<size_t>(INPUT_CHANNELS));
    const auto workspace_size = 
                   convolve_3::get_workspace_size(input_channels, output_channels);
    const auto winograd_V_size = workspace_size.first;
    const auto winograd_M_size = workspace_size.second;


    auto winograd_V = std::vector<float>(winograd_V_size);
    auto winograd_M = std::vector<float>(winograd_M_size);


    auto conv_out = std::vector<float>(output_channels * intersections);
    auto conv_in = std::vector<float>(output_channels * intersections);
    auto res = std::vector<float>(output_channels * intersections);

    input_channels = INPUT_CHANNELS;

    convolve_3::Forward(input_channels, output_channels, planes,
                        m_weights->input_conv.weights, 
                        winograd_V, winograd_M, conv_out);


    batchnorm::Forward(output_channels, conv_out,
                       m_weights->input_bn.means,
                       m_weights->input_bn.stddevs,
                       nullptr, false);

    inputpool::Forward(INPUT_FEATURES, output_channels,
                       features,
                       m_weights->input_fc.weights,
                       m_weights->input_fc.biases,
                       conv_out);

    input_channels = m_weights->channels;

    // residual tower

    const size_t residuals =  m_weights->residuals;
    for (auto i = size_t{0}; i < residuals; ++i) {
        const auto tower_channels = m_weights->channels;
        const auto tower_ptr = m_weights->residual_tower.data() + i;

        std::swap(conv_in, conv_out);
        convolve_3::Forward(input_channels, tower_channels, conv_in,
                            tower_ptr->conv_1.weights,
                            winograd_V, winograd_M, conv_out);

        batchnorm::Forward(tower_channels, conv_out,
                           tower_ptr->bn_1.means,
                           tower_ptr->bn_1.stddevs);

        std::swap(conv_in, res);
        std::swap(conv_out, conv_in);
        convolve_3::Forward(input_channels, tower_channels, conv_in,
                            tower_ptr->conv_2.weights,
                            winograd_V, winograd_M, conv_out);

        batchnorm::Forward(tower_channels, conv_out,
                           tower_ptr->bn_2.means,
                           tower_ptr->bn_2.stddevs,
                           nullptr, false);

        const size_t se_size = 4 * tower_channels;
        se_unit::Forward(tower_channels, se_size,
                         conv_out, res, 
                         tower_ptr->extend.weights,
                         tower_ptr->extend.biases,
                         tower_ptr->squeeze.weights,
                         tower_ptr->squeeze.biases);
    }

    // policy head
    auto policy_conv = std::vector<float>(OUTPUTS_POLICY * intersections);
    auto policy_pool = std::vector<float>(OUTPUTS_POLICY);
    auto pass_out = std::vector<float>(1);

    convolve_1::Forward(input_channels, OUTPUTS_POLICY, conv_out, 
                        m_weights->p_conv.weights,
                        policy_conv);

    batchnorm::Forward(OUTPUTS_POLICY, policy_conv, 
                       m_weights->p_bn.means,
                       m_weights->p_bn.stddevs);

    convolve_1::Forward(OUTPUTS_POLICY, OUTPUTS_PRBAOBILITIES, policy_conv, 
                        m_weights->prob_conv.weights,
                        output_pol);

    globalpool::Forward(OUTPUTS_POLICY,
                        policy_conv,
                        policy_pool);


    FullyConnect::Forward(OUTPUTS_POLICY, OUTPUTS_PASS,
                          policy_pool, 
                          m_weights->pass_fc.weights,
                          m_weights->pass_fc.biases, 
                          pass_out, false);

    // probabilities
    output_pol[intersections] = pass_out[0];

    // value head
    auto value_conv = std::vector<float>(OUTPUTS_VALUE * intersections);
    auto value_pool = std::vector<float>(OUTPUTS_VALUE);

    convolve_1::Forward(input_channels, OUTPUTS_VALUE, conv_out, 
                        m_weights->v_conv.weights,
                        value_conv);

    batchnorm::Forward(OUTPUTS_VALUE, value_conv, 
                       m_weights->v_bn.means,
                       m_weights->v_bn.stddevs);
    // score belief
    convolve_1::Forward(OUTPUTS_VALUE, OUTPUTS_SCOREBELIEF, value_conv, 
                      m_weights->sb_conv.weights,
                      output_sb);
    // ownership
    convolve_1::Forward(OUTPUTS_VALUE, OUTPUTS_OWNERSHIP, value_conv, 
                        m_weights->os_conv.weights,
                        output_os);

    globalpool::Forward(OUTPUTS_VALUE,
                        value_conv,
                        value_pool);

    // final score
    FullyConnect::Forward(OUTPUTS_VALUE, FINAL_SCORE,
                          value_pool, 
                          m_weights->fs_fc.weights,
                          m_weights->fs_fc.biases, 
                          output_fs, false);

    // winrate misc
    FullyConnect::Forward(OUTPUTS_VALUE, VALUE_MISC,
                          value_pool, 
                          m_weights->v_fc.weights,
                          m_weights->v_fc.biases, 
                          output_val, false);
}
};

#define CASE_PIPE(BSIZE)                               \
case BSIZE:                                            \
    {                                                  \
        auto pipe = FORWARD_PIPE<BSIZE>();             \
        pipe.forward(m_weights, planes, features,      \
                    output_pol, output_sb,             \
                    output_os, output_fs, output_val); \
    }                                                  \
    break; 

void CPUbackend::initialize(std::shared_ptr<Model::NNweights> weights) {
    m_weights = weights;
    Model::winograd_transform(m_weights);
}

void CPUbackend::reload(std::shared_ptr<Model::NNweights> weights) {
    if (m_weights != nullptr) {
        m_weights.reset();
    }
    m_weights = weights;
    Model::winograd_transform(m_weights);
}

void CPUbackend::forward(const int boardsize,
                         const std::vector<float> &planes,
                         const std::vector<float> &features,
                         std::vector<float> &output_pol,
                         std::vector<float> &output_sb,
                         std::vector<float> &output_os,
                         std::vector<float> &output_fs,
                         std::vector<float> &output_val) {

    switch (boardsize) {
        CASE_PIPE(2);
        CASE_PIPE(3);
        CASE_PIPE(4);
        CASE_PIPE(5);
        CASE_PIPE(6);
        CASE_PIPE(7);
        CASE_PIPE(8);
        CASE_PIPE(9);
        CASE_PIPE(10);
        CASE_PIPE(11);
        CASE_PIPE(12);
        CASE_PIPE(13);
        CASE_PIPE(14);
        CASE_PIPE(15);
        CASE_PIPE(16);
        CASE_PIPE(17);
        CASE_PIPE(18);
        CASE_PIPE(19);
        CASE_PIPE(20);
        CASE_PIPE(21);
        CASE_PIPE(22);
        CASE_PIPE(23);
        CASE_PIPE(24);
        CASE_PIPE(25);
        default: 
            Utils::auto_printf("Not support for %d x %d board\n", boardsize, boardsize);
            break;
    }
}

void CPUbackend::release() {
    if (m_weights != nullptr) {
        m_weights.reset();
    }
    m_weights = nullptr;
}

bool CPUbackend::valid() {
    return m_weights->loaded;
}
