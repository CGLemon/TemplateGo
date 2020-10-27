#ifndef MODEL_H_INCLUDE
#define MODEL_H_INCLUDE

#include <vector>
#include <memory>
#include <iostream>
#include <fstream>
#include <sstream>

#include "Cache.h"
#include "GameState.h"
#include "config.h"
#include "Blas.h"
#include "Utils.h"

static constexpr auto INPUT_CHANNELS = 24;
static constexpr auto INPUT_FEATURES = 10;

static constexpr auto OUTPUTS_POLICY = 32;
static constexpr auto OUTPUTS_PRBAOBILITIES = 1;
static constexpr auto OUTPUTS_PASS = 1;

static constexpr auto OUTPUTS_VALUE = 64;
static constexpr auto OUTPUTS_SCOREBELIEF = 2;
static constexpr auto OUTPUTS_OWNERSHIP = 1;

static constexpr auto FINAL_SCORE = 1;
static constexpr auto VALUE_MISC = 2;
// static constexpr auto VALUE_LABELS = 21;
// static constexpr auto LABELS_CENTER = 10;
static constexpr auto POTENTIAL_MOVES = NUM_INTERSECTIONS + 1;

struct Desc {
    struct ConvLayer {
        void load_weights(std::vector<float> &loadweights);
        std::vector<float> weights;
    };

    struct BatchNormLayer {
        void load_means(std::vector<float> &loadweights);
        void load_stddevs(std::vector<float> &loadweights);
        std::vector<float> means;
        std::vector<float> stddevs;
    };

    struct LinearLayer {
        void load_weights(std::vector<float> &loadweights);
        void load_biases(std::vector<float> &loadweights);
        std::vector<float> weights;
        std::vector<float> biases;
    };
};


struct Model {
    struct NNweights {

        struct ResidualTower {
            Desc::ConvLayer conv_1;
            Desc::BatchNormLayer bn_1;
            Desc::ConvLayer conv_2;
            Desc::BatchNormLayer bn_2;

            Desc::LinearLayer extend;
            Desc::LinearLayer squeeze;
        };

        bool loaded{false};
        size_t channels{0};
        size_t residuals{0};
    

        // input layer
        Desc::ConvLayer input_conv;
        Desc::BatchNormLayer input_bn;
        Desc::LinearLayer input_fc;

        // residual tower
        std::vector<ResidualTower> residual_tower;

        // policy head
        Desc::ConvLayer p_conv;
        Desc::BatchNormLayer p_bn;

        Desc::ConvLayer prob_conv;     // probability
        Desc::LinearLayer pass_fc;     // pass

        // value head
        Desc::ConvLayer v_conv;
        Desc::BatchNormLayer v_bn;

        Desc::ConvLayer sb_conv;
        Desc::ConvLayer os_conv;
        Desc::LinearLayer fs_fc;
        Desc::LinearLayer v_fc;

    };


    class NNpipe {
    public:
        virtual void initialize(std::shared_ptr<NNweights> weights) = 0;
        virtual void forward(const int boardsize,
                             const std::vector<float> &planes,
                             const std::vector<float> &features,
                             std::vector<float> &output_pol,
                             std::vector<float> &output_sb,
                             std::vector<float> &output_os,
                             std::vector<float> &output_fs,
                             std::vector<float> &output_val) = 0;

        virtual void reload(std::shared_ptr<Model::NNweights> weights) = 0;
        virtual void release() = 0;
        virtual void destroy() = 0;
        virtual bool valid() = 0;
    };

    static void loader(const std::string &filename,
                       std::shared_ptr<NNweights> &nn_weight);
    static void fill_weights(std::istream &weights_file,
                             std::shared_ptr<NNweights> &nn_weight);

    static std::vector<float> gather_planes(const GameState *const state, 
                                            const int symmetry);

    static std::vector<float> gather_features(const GameState *const state);

    static void features_stream(std::ostream &out,
                                const GameState *const state,
                                const int symmetry);

    static std::string features_to_string(GameState &state, const int symmetry);

    static NNResult get_result(const GameState *const state,
                               std::vector<float> &policy,
                               std::vector<float> &score_belief,
                               std::vector<float> &ownership,
                               std::vector<float> &final_score,
                               std::vector<float> &values,
                               const float softmax_temp,
                               const int symmetry);

    static float get_winrate(GameState &state, const NNResult &result);
    static float get_winrate(GameState &state, const NNResult &result, float current_komi);

    static void winograd_transform(std::shared_ptr<NNweights> &nn_weight);

    static void fill_fullyconnect_layer(Desc::LinearLayer &layer, std::istream &weights_file);

    static void fill_batchnorm_layer(Desc::BatchNormLayer &layer, std::istream &weights_file);

    static void fill_convolution_layer(Desc::ConvLayer &layer, std::istream &weights_file);
};



#endif
