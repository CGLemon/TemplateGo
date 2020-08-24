#include "cfg.h"
#include "config.h"
#include "gtp.h"
#include "Utils.h"

#include <string>
#include <sstream>
#include <fstream>
#include <iostream>

bool cfg_quiet;
FILE *cfg_logfile_stream;
bool cfg_allowed_suicide;
bool cfg_pre_block_superko;
bool cfg_gtp_mode;

float cfg_softmax_temp;
int cfg_cache_moves;
float cfg_cache_ratio;

std::string cfg_weightsfile;
int cfg_uct_threads;
float cfg_fpu_root_reduction;
float cfg_fpu_reduction;
float cfg_logpuct;
float cfg_puct;
float cfg_logconst;
float cfg_delta_attenuation_ratio;
bool cfg_dirichlet_noise;

int cfg_maintime;
int cfg_byotime;
int cfg_byostones;

bool cfg_auto_quit;
std::string cfg_selfplay_config;
bool cfg_selfplay_agent;
size_t cfg_random_move_cnt;
bool cfg_random_move;
int cfg_boardsize;
float cfg_komi;
int cfg_playouts;
float cfg_allow_pass_ratio;
int cfg_batchsize;
bool cfg_ponder;
size_t cfg_random_min_visits;

float cfg_resign_threshold;

size_t cfg_max_game_buffer;
bool cfg_collect;

std::uint64_t cfg_default_seed;

void dump_setting() {
  Utils::auto_printf("Setting\n");
  Utils::auto_printf(" batch size %d\n", cfg_batchsize);
  Utils::auto_printf(" threads %d\n", cfg_uct_threads);
}


void arg_parser(int argc, char **argv) {

  bool success = true;
  auto warning_stream = std::stringstream{""};
  warning_stream << "Warning!" << std::endl;

  for (auto i = int{1}; i < argc; ++i) {
    std::string cmd(argv[i]);
    if (cmd == "-g" || cmd == "--gtp") {
      gtp::gtp_mode();
    } else if (cmd == "-w" || cmd == "--weight") {
      if ((i + 1) <= argc && argv[(i + 1)][0] != '-') {
        i++;
        std::string filename(argv[i]);
        cfg_weightsfile = filename;
      }
    } else if (cmd == "-p" || cmd == "--playouts") {
      if ((i + 1) <= argc && argv[(i + 1)][0] != '-' ) {
        i++;
        std::string numplayouts(argv[i]);
        if (Utils::is_allnumber(numplayouts)) {
          cfg_playouts = std::stoi(numplayouts);
        } else {
          success = false;
          warning_stream << cmd << " must be integer.\n";
        }
      }
    } else if (cmd == "-t" || cmd == "--thread") {
      if ((i + 1) <= argc && argv[(i + 1)][0] != '-') {
        i++;
        std::string numthreads(argv[i]);
        if (Utils::is_allnumber(numthreads)) {
          cfg_uct_threads = std::stoi(numthreads);
        } else {
          success = false;
          warning_stream << cmd << " must be integer.\n";
        }
      }
    } else if (cmd == "--batchsize") {
      if ((i + 1) <= argc && argv[(i + 1)][0] != '-') {
        i++;
        std::string numbatch(argv[i]);
        if (Utils::is_allnumber(numbatch)) {
          cfg_batchsize = std::stoi(numbatch);
        } else {
          success = false;
          warning_stream << cmd << " must be integer.\n";
        }
      }
    } else if (cmd == "--no_resign") {
      cfg_resign_threshold = -1.f;
    } else if (cmd == "--noise") {
      cfg_dirichlet_noise = true;
    } else if (cmd == "--ponder") {
      cfg_ponder = true;
    } else if (cmd == "--random_move") {
      cfg_random_move = true;
    } else if (cmd == "--collect") {
      cfg_collect = true;
    } else {
      success = false;
      warning_stream << cmd << " not understood\n";
    }
  }

  if (!success) {
    Utils::auto_printf("%s\n", warning_stream.str().c_str());
  }

  adjust_batchsize(cfg_batchsize);
  dump_setting();
}

void init_cfg() {
  cfg_quiet = false;
  cfg_logfile_stream = nullptr;
  cfg_allowed_suicide = false;
  cfg_gtp_mode = false;
  cfg_softmax_temp = 1.0f;
  cfg_cache_moves = 5;
  cfg_cache_ratio = 0.2f;

  cfg_weightsfile = "_NO_WEIGHTS_FILE_";
  cfg_uct_threads = 1;
  cfg_fpu_root_reduction = 0.25f;
  cfg_fpu_reduction = 0.25f;
  cfg_logpuct = 0.015f;
  cfg_puct = 0.5f;
  cfg_logconst = 1.7f;
  cfg_dirichlet_noise = false;
  cfg_ponder = false;

  cfg_maintime = 60 * 60 * 1;
  cfg_byotime = 0;
  cfg_byostones = 0;

  cfg_auto_quit = false;

  cfg_selfplay_config = "_NO_CONFIG_FILE_";
  cfg_selfplay_agent = false;
  cfg_random_move_cnt = 0;
  cfg_random_move = false;
  cfg_boardsize = DEFULT_BOARDSIZE;
  cfg_komi = DEFULT_KOMI;
  cfg_playouts = 1600;
  cfg_random_min_visits = 1;
  cfg_allow_pass_ratio = 0.5f;
  cfg_batchsize = 1;
  
  cfg_resign_threshold = 0.05f;

  cfg_max_game_buffer = 9999;
  cfg_collect = false;

  cfg_default_seed = Utils::rng_seed();
}

void adjust_batchsize(int batchsize) {
  if (cfg_uct_threads < batchsize) {
    cfg_batchsize = cfg_uct_threads;
  }
}
