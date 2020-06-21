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

int cfg_boardsize;
float cfg_komi;
int cfg_playouts;
float cfg_allow_pass_ratio;


void arg_parser(int argc, char **argv) {

  for (int i = 1; i < argc; ++i) {
    std::string cmd(argv[i]);
    if (cmd == "-g" || cmd == "--gtp") {
      gtp::gtp_mode();
      Utils::auto_printf("GTP mode!\n");
    } else if (cmd == "-w" || cmd == "--weight") {
      if ((i + 1) <= argc) {
        std::string filename(argv[(i + 1)]);
        cfg_weightsfile = filename;
      }
    } else if (cmd == "-p" || cmd == "--playouts") {
      if ((i + 1) <= argc) {
        std::string numplayouts(argv[(i + 1)]);
        if (Utils::is_allnumber(numplayouts))
          cfg_playouts = std::stoi(numplayouts);
        else
          Utils::auto_printf("%s must be integer.\n", cmd.c_str());
      }
    } else if (cmd == "-t" || cmd == "--thread") {
      if ((i + 1) <= argc) {
        std::string numthreads(argv[(i + 1)]);
        if (Utils::is_allnumber(numthreads))
          cfg_uct_threads = std::stoi(numthreads);
        else
          Utils::auto_printf("%s must be integer.\n", cmd.c_str());
      }

    }
  }
}

void init_cfg() {
  cfg_quiet = false;
  cfg_logfile_stream = nullptr;
  cfg_allowed_suicide = false;
  cfg_gtp_mode = false;
  cfg_softmax_temp = 1.0f;
  cfg_cache_moves = 5;

  cfg_weightsfile = "NO_WEIGHT_FILE";
  cfg_uct_threads = 0;
  cfg_fpu_root_reduction = 0.25f;
  cfg_fpu_reduction = 0.25f;
  cfg_logpuct = 0.015f;
  cfg_puct = 0.5f;
  cfg_logconst = 1.7f;
  cfg_delta_attenuation_ratio = 1.0f;
  cfg_dirichlet_noise = false;

  cfg_maintime = 60 * 60 * 100;
  cfg_byotime = 0;
  cfg_byostones = 0;

  cfg_boardsize = DEFULT_BOARDSIZE;
  cfg_komi = DEFULT_KOMI;
  cfg_playouts = 1600;
  cfg_allow_pass_ratio = 0.2f;
}

//TODO: 用外部文件來設定 cfg
bool cfg_loader_parser(std::string cfg) {
  return false;
}

void cfg_loader(std::string filename) {
  std::ifstream cfg_file(filename.c_str());

  auto stream_line = std::string{};
  while (std::getline(cfg_file, stream_line)) {
    if(!cfg_loader_parser(stream_line)) break;
  }
  cfg_file.close();
}
