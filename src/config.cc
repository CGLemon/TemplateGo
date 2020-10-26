#include "config.h"
#include "Zobrist.h"
#include "Utils.h"

#include <stdexcept>
#include <string>
#include <mutex>

std::unordered_map<std::string, Utils::Option> options_map;
// std::mutex map_mutex;

#define OPTIONS_EXPASSION(T)                  \
template<>                                    \
T option<T>(std::string name) {               \
    auto res = options_map.find(name);        \
    if (res == std::end(options_map)) {       \
        auto out = std::string{"Not Found "}; \
         out += name;                         \
        throw std::runtime_error(out);        \
    }                                         \
    return res->second.get<T>();              \
}

OPTIONS_EXPASSION(std::string)
OPTIONS_EXPASSION(bool)
OPTIONS_EXPASSION(int)
OPTIONS_EXPASSION(float)

#undef OPTIONS_EXPASSION

#define OPTIONS_SET_EXPASSION(T)                     \
template<>                                           \
bool set_option<T>(std::string name, T val) {        \
    auto res = options_map.find(name);               \
    if (res != std::end(options_map)) {              \
        res->second.set<T>(val);                     \
        return true;                                 \
    }                                                \
    return false;                                    \
}

OPTIONS_SET_EXPASSION(std::string)
OPTIONS_SET_EXPASSION(bool)
OPTIONS_SET_EXPASSION(int)
OPTIONS_SET_EXPASSION(float)

#undef OPTIONS_SET_EXPASSION

void init_options_map() {

    // basic options
    options_map["name"] << Utils::Option::setoption(PROGRAM);
    options_map["version"] << Utils::Option::setoption(VERSION);

    options_map["mode"] << Utils::Option::setoption(std::string{"ascii"});
    options_map["help"] << Utils::Option::setoption(false);

    options_map["quiet"] << Utils::Option::setoption(false);
    options_map["num_games"] << Utils::Option::setoption(1, 32, 1);
    options_map["reserve_movelist"] << Utils::Option::setoption(60);
    options_map["threads"] << Utils::Option::setoption(1, 256, 1);

    // rules
    options_map["allow_suicide"] << Utils::Option::setoption(false);
    options_map["pre_block_superko"] << Utils::Option::setoption(false);
    options_map["boardsize"] << Utils::Option::setoption(BOARD_SIZE, MARCO_BOARD_SIZE, MARCO_MINIMAL_GTP_BOARD_SIZE);
    options_map["komi"] << Utils::Option::setoption(DEFAULT_KOMI, MARCO_MAXIMAL_KOMI, MARCO_MINIMAL_KOMI);

    // io
    options_map["float_precision"] << Utils::Option::setoption(5);

    // network default parameters
    options_map["weights_file"] << Utils::Option::setoption(std::string{"_NO_FILE_"});
    options_map["softmax_temp"] << Utils::Option::setoption(1.0f);
    options_map["cache_moves"] << Utils::Option::setoption(25);
    // options_map["mutil_labeled_komi"] << Utils::Option::setoption(0, 10, -10);
    options_map["batchsize"] << Utils::Option::setoption(1, 32, 1);
    options_map["waittime"] << Utils::Option::setoption(10);

    // uct search
    options_map["resigned_threshold"] << Utils::Option::setoption(0.1f, 1, 0);
    options_map["allowed_pass_ratio"] << Utils::Option::setoption(0.8f, 1, 0);
    options_map["playouts"] << Utils::Option::setoption(1600);
    options_map["dirichlet_noise"] << Utils::Option::setoption(false);
    options_map["fpu_root_reduction"] << Utils::Option::setoption(0.25f);
    options_map["fpu_reduction"] << Utils::Option::setoption(0.25f);
    options_map["logpuct"] << Utils::Option::setoption(0.015f);
    options_map["logconst"] << Utils::Option::setoption(1.7f);
    options_map["puct"] << Utils::Option::setoption(0.5f);
    options_map["score_utility_div"] << Utils::Option::setoption(3.5f);
    options_map["ponder"] << Utils::Option::setoption(false);
    options_map["random_min_visits"] << Utils::Option::setoption(1);

    // time control paramters
    options_map["maintime"] << Utils::Option::setoption(3600);
    options_map["byotime"] << Utils::Option::setoption(0);
    options_map["byostones"] << Utils::Option::setoption(0);
    options_map["lagbuffer"] << Utils::Option::setoption(100.f);

    // trainer
    options_map["collect"] << Utils::Option::setoption(false); 
    options_map["max_game_buffer"] << Utils::Option::setoption(1000);

    // self-play
    options_map["random_move"] << Utils::Option::setoption(false);
    options_map["random_move_div"] << Utils::Option::setoption(1);
}

void init_basic_parameters() {

    Zobrist::init_zobrist();
    init_options_map();
}


ArgsParser::ArgsParser(int argc, char** argv) {

    auto parser = Utils::CommandParser(argc, argv);

    const auto is_parameter = [](const std::string &para) -> bool {
        if (para.empty()) {
            return false;
        }
        return para[0] != '-';
    };

    using List = std::vector<std::string>;

    if (const auto res = parser.find(List{"--help", "-h"})) {
        set_option("help", true);
    }

    if (const auto res = parser.find("--noise")) {
        set_option("dirichlet_noise", true);
    }

    if (const auto res = parser.find("--random")) {
        set_option("random_move", true);
    }

    if (const auto res = parser.find_next("--random_div")) {
        if (is_parameter(res->str)) {
            set_option("random_move_div", res->get<int>());
        }
    }

    if (const auto res = parser.find_next(List{"--weights", "-w"})) {
        if (is_parameter(res->str)) {
            set_option("weights_file", res->str);
        }
    }

    if (const auto res = parser.find_next(List{"--playouts", "-p"})) {
        if (is_parameter(res->str)) {
            set_option("playouts", res->get<int>());
        }
    }

    if (const auto res = parser.find_next(List{"--threads", "-t"})) {
        if (is_parameter(res->str)) {
            set_option("threads", res->get<int>());
        }
    }

    if (const auto res = parser.find_next("--boardsize")) {
        if (is_parameter(res->str)) {
            set_option("boardsize", res->get<int>());
        }
    }

    if (const auto res = parser.find_next(List{"--batchsize", "-b"})) {
        if (is_parameter(res->str)) {
            set_option("batchsize", res->get<int>());
        }
    }

    if (const auto res = parser.find_next("--komi")) {
        if (is_parameter(res->str)) {
            set_option("komi", res->get<float>());
        }
    }

    if (const auto res = parser.find_next(List{"--mode", "-m"})) {
        if (is_parameter(res->str)) {
            if (res->str == "ascii" || res->str == "gtp" || res->str == "selfplay") {
                set_option("mode", res->str);
            } else {
                Utils::auto_printf("syntax not understood : %s\n", res->get<const char*>());
            }
        }
    }

    if (const auto res = parser.find_next("--resigned")) {
        if (is_parameter(res->str)) {
            set_option("resigned_threshold", res->get<float>());
        }
    }

    if (const auto res = parser.find("--collect")) {
        set_option("collect", true);
    }

    if (option<std::string>("mode") == "gtp") {
        set_option("quiet", true);
    }

    if (option<std::string>("mode") == "selfplay") {
        set_option("collect", true);
        set_option("quiet", true);
    }
}

void ArgsParser::help() const {
    Utils::auto_printf("Argumnet\n");
    Utils::auto_printf(" --help, -h\n");
    Utils::auto_printf(" --mode, -m [ascii/gtp]\n");
    Utils::auto_printf(" --playouts, -p <integral>\n");
    Utils::auto_printf(" --threads, -t <integral>\n");
    Utils::auto_printf(" --weights, -w <weights file>\n");
    Utils::auto_printf(" --komi <float>\n");
    Utils::auto_printf(" --boardsize <integral>\n");
    Utils::auto_printf(" --batchsize, -b <integral>\n");
}

void ArgsParser::dump() const {
    if (option<bool>("help")) {
        help();
    }
    Utils::auto_printf("Threads : %d\n", option<int>("threads"));
    Utils::auto_printf("Batchsize : %d\n", option<int>("batchsize"));
}

