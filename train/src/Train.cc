#include <iostream>
#include <string>
#include <vector>
#include <sstream>
#include <random>
#include <iomanip>

#include "Loader.h"
#include "Symmetry.h"
#include "Torch.h"
/*
  example 1:
  --datafiles 1 test.txt --boardSize 9 --residualBlocks 2 --residualChannels 32 --iterate 200 --batchSize 64 --learningRate 0.001 --weightDecay 0.001 --weightsName weights.txt --CUP_only

  example 2:
  --boardSize 9  --residualBlocks 2 --residualChannels 32 --weightsName weights.txt

*/

constexpr static int DEFULT_INPUT_CHANNELS = 24;
constexpr static int DEFULT_INPUT_FEATURES = 10;

struct ArgsParser {

    ArgsParser(int argc, char **argv) {
        for (int i = 1; i < argc; ++i) {
            auto cmd = std::string{argv[i]};
            if (cmd == "--datafiles") {
                if (check_next(i, argc, argv)) {
                    i++;
                } else {
                    continue;
                }

                int cnt = std::stoi(std::string{argv[i]});

                for (int c = 0; c < cnt; ++c) {
                    if (check_next(i, argc, argv)) {
                        i++;
                        auto fname = std::string{argv[i]};
                        data_names.emplace_back(fname);
                    }
                }
            } else if (cmd == "--inputChannels") {
                if (check_next(i, argc, argv)) {
                    i++;
                    input_channels = std::stoi(std::string{argv[i]});
                }
            } else if (cmd == "--inputFeatures") {
                if (check_next(i, argc, argv)) {
                    i++;
                    input_features = std::stoi(std::string{argv[i]});
                }
            } else if (cmd == "--residualChannels") {
                if (check_next(i, argc, argv)) {
                    i++;
                    residual_channels = std::stoi(std::string{argv[i]});
                }
            } else if (cmd == "--residualBlocks")  {
                if (check_next(i, argc, argv)) {
                    i++;
                    residual_blocks = std::stoi(std::string{argv[i]});
                }
            } else if (cmd == "--iterate")  {
                if (check_next(i, argc, argv)) {
                    i++;
                    iterate = std::stoi(std::string{argv[i]});
                }
            } else if (cmd == "--boardSize")  {
                if (check_next(i, argc, argv)) {
                    i++;
                    boardsize = std::stoi(std::string{argv[i]});
                }
            } else if (cmd == "--batchSize")  {
                if (check_next(i, argc, argv)) {
                    i++;
                    batch_size = std::stoi(std::string{argv[i]});
                }
            } else if (cmd == "--learningRate")  {
                if (check_next(i, argc, argv)) {
                    i++;
                    learning_rate = std::stof(std::string{argv[i]});
                }
            } else if (cmd == "--weightDecay")  {
                if (check_next(i, argc, argv)) {
                    i++;
                    weight_decay = std::stof(std::string{argv[i]});
                }
            } else if (cmd == "--weightsName")  {
                if (check_next(i, argc, argv)) {
                    i++;
                    weights_name = std::string{argv[i]};
                }
            } else if (cmd == "--dumpStep") {
                if (check_next(i, argc, argv)) {
                    i++;
                    dump_step = std::stoi(std::string{argv[i]});
                }
            } else if (cmd == "--CUP_only") {
                force_cpu = true;
            }
        }
    }

    bool check_next(int idx, int argc, char **argv) {
        if (idx + 1 < argc && argv[idx+1][0] != '-') {
            return true;
        }
        return false;
    }

    void dump_setting() {
        auto out = std::ostringstream{};
        setting_stream(out);
        std::cout << out.str();
    }

    void setting_stream(std::ostream &out) {

        const int remain = 18;

        out << std::setw(remain) << " Input data" << " << ";
        if (data_names.empty()) {
            out << "_NO_DATA_FILES_";
        } else {
            for (auto &dname : data_names) {
                out << dname << " ";
            }
        }
        out << std::setprecision(5) << std::endl;
        out << std::setw(remain) << " Board size"        << " : " << boardsize          << std::endl;
        out << std::setw(remain) << " Input channels"    << " : " << input_channels     << std::endl;
        out << std::setw(remain) << " Input features"    << " : " << input_features     << std::endl;
        out << std::setw(remain) << " Residual channels" << " : " << residual_channels  << std::endl;
        out << std::setw(remain) << " Residual blocks"   << " : " << residual_blocks    << std::endl;
        out << std::setw(remain) << " Iterate"           << " : " << iterate            << std::endl;
        out << std::setw(remain) << " Batch size"        << " : " << batch_size         << std::endl;
        out << std::setw(remain) << " Learning rate"     << " : " << learning_rate      << std::endl;
        out << std::setw(remain) << " Weight decay"      << " : " << weight_decay       << std::endl;
        out << std::setw(remain) << " Weights name"      << " : " << weights_name       << std::endl;
        out << std::setw(remain) << " Dump step"         << " : " << dump_step          << std::endl;

        if (force_cpu)
            out << std::setw(remain) << " CPU only" << " : TRUE" << std::endl;

        out << std::setprecision(6); // back to default setting
    }

    std::vector<std::string> data_names;
    int input_channels{DEFULT_INPUT_CHANNELS};
    int input_features{DEFULT_INPUT_FEATURES};
    int residual_channels{0};
    int residual_blocks{0};
    int iterate{0};
    int boardsize{0};
    int batch_size{0};

    bool force_cpu{false};
    float learning_rate{0.0f};
    float weight_decay{0.0f};
    int dump_step{1000};
    std::string weights_name{"_NO_WEIGHT_FILE_NAME_"};
};

void training(ArgsParser &args, Loader &loader) {
#ifdef USE_TORCH
    const auto buffer = loader.get_buffer();
    auto symmetry = Symmetry((size_t)args.boardsize);
    auto train_config = Torch::TrainConfig{};

    train_config.in_channels = args.input_channels;
    train_config.in_features = args.input_features;
    train_config.resnet_channels = args.residual_channels;
    train_config.resnet_blocks = args.residual_blocks;
    train_config.boardsize = args.boardsize;
    train_config.learning_rate = args.learning_rate;
    train_config.weight_decay = args.weight_decay;
    train_config.force_cpu = args.force_cpu;

    auto train_helper = std::make_shared<Torch::Train_helper>();
    train_helper->init(train_config);
    std::random_device rd;
    std::mt19937 gen = std::mt19937(rd());
    std::uniform_int_distribution<int> dis(Symmetry::IDENTITY_SYMMETRY, (Symmetry::NUM_SYMMETRIES-1));

    const auto batchsize = args.batch_size;
    const auto iterate = args.iterate;
    const auto boardsize = args.boardsize;
    const auto intersections = boardsize * boardsize;
    const auto in_channels = args.input_channels;
    const auto in_features = args.input_features;

    auto backword_cnt = size_t{0};
    for (int ite = 0; ite < iterate; ++ite) {
        auto end = bool{false};
        auto buffer_ptr = std::begin(buffer);
        while (!end) {
            auto batches = std::vector<Torch::TrainDataBuffer>{};
            for (int b = 0; b < batchsize; ++b) {
                const auto sym = dis(gen);
                batches.emplace_back(Torch::TrainDataBuffer{});

                const size_t in_planes_size = (*buffer_ptr)->input_planes.size();
                const size_t in_features_size = (*buffer_ptr)->input_features.size();
                const size_t probabilities_size = (*buffer_ptr)->probabilities.size();
                const size_t ownership_size = (*buffer_ptr)->ownership.size(); 
                const size_t winrate_size = (*buffer_ptr)->winrate.size();

                batches[b].input_planes = std::vector<float>(in_planes_size);
                batches[b].input_features = std::vector<float>(in_features_size);
                batches[b].probabilities = std::vector<float>(probabilities_size);
                batches[b].opponent_probabilities = std::vector<float>(probabilities_size);
                batches[b].ownership = std::vector<float>(ownership_size);
                batches[b].winrate = std::vector<float>(winrate_size);

                for (int c = 0; c < in_channels; ++c) {
                    for (int idx = 0; idx < intersections; ++idx) {
                        const int sym_idx = symmetry.get_transform_idx(idx, sym);
                        const int start = c * intersections;
                        batches[b].input_planes[start + sym_idx] = (*buffer_ptr)->input_planes[start + idx];
                    }
                }

                for (int idx = 0; idx < in_features; ++idx) {
                    batches[b].input_features[idx] = (*buffer_ptr)->input_features[idx];
                }

                for (int idx = 0; idx < intersections; ++idx) {
                    const int sym_idx = symmetry.get_transform_idx(idx, sym);
                    batches[b].probabilities[sym_idx] = (*buffer_ptr)->probabilities[idx];   
                }
                batches[b].probabilities[intersections] = (*buffer_ptr)->probabilities[intersections]; // pass

                for (int idx = 0; idx < intersections; ++idx) {
                    const int sym_idx = symmetry.get_transform_idx(idx, sym);
                    batches[b].opponent_probabilities[sym_idx] = (*buffer_ptr)->opponent_probabilities[idx];   
                }
                batches[b].opponent_probabilities[intersections] = (*buffer_ptr)->opponent_probabilities[intersections]; // pass

                batches[b].scorebelief_idx = (*buffer_ptr)->scorebelief_idx;
                batches[b].final_score = (*buffer_ptr)->final_score;
                                                      
                for (int idx = 0; idx < intersections; ++idx) {
                    const int sym_idx = symmetry.get_transform_idx(idx, sym);
                    batches[b].ownership[sym_idx] = (*buffer_ptr)->ownership[idx];
                }

                batches[b].current_komi = (*buffer_ptr)->current_komi;


                for (int idx = 0; idx < winrate_size; ++idx) {
                    batches[b].winrate[idx] = (*buffer_ptr)->winrate[idx];         
                }

                buffer_ptr++;
                if (buffer_ptr == std::end(buffer)) {
                    end = true;
                    break;
                }
            }
            backword_cnt++;
            const auto loss = train_helper->train_batch(batches);
            if (backword_cnt % args.dump_step == 0) {
                printf("========================== step : %zu ==========================\n", backword_cnt);
                for (auto &l : loss) {
                    const auto name = l.first;
                    const auto num = l.second;
                    printf("%s >> %f\n", name.c_str(), num);
                }
            }
        }
    }
    printf("Save the weights file %s\n", args.weights_name.c_str());
    train_helper->save_weights(args.weights_name);

#else
    printf("Need libtorch\n");

#endif
}

void create_raw_network(ArgsParser &args) {
#ifdef USE_TORCH
    auto train_config = Torch::TrainConfig{};
    train_config.in_channels = args.input_channels;
    train_config.in_features = args.input_features;
    train_config.resnet_channels = args.residual_channels;
    train_config.resnet_blocks = args.residual_blocks;
    train_config.boardsize = args.boardsize;
    train_config.learning_rate = args.learning_rate;
    train_config.weight_decay = args.weight_decay;
    train_config.force_cpu = true;

    auto train_helper = std::make_shared<Torch::Train_helper>();
    train_helper->init(train_config);
    train_helper->save_weights(args.weights_name);

    printf(" Success to create a raw weights file %s\n", args.weights_name.c_str());

#else
    printf("Need libtorch\n");

#endif
}


int main(int argc, char **argv) {
    auto args = ArgsParser(argc, argv);
    args.dump_setting();
    std::cout << std::endl;

    auto loader = Loader();
    loader.set_size(args.boardsize, args.input_channels, args.input_features);
    loader.load_data_from_filenames(args.data_names);
    loader.dump_memory();
    std::cout << std::endl;

    if (args.iterate == 0 || loader.get_buffer().empty()) {
        create_raw_network(args);
    } else {
        std::cout << "Start trainig..." << std::endl;
        training(args, loader);
        std::cout << "End trainig..." << std::endl;
    }
    return 0;
}
