#include "cfg.h"
#include "config.h"
#include "gtp.h"

#include <string>

bool cfg_quiet;
FILE * cfg_logfile_stream;
bool cfg_allowed_suicide;
bool cfg_pre_block_superko;
bool cfg_gtp_mode;


float cfg_softmax_temp;
int cfg_cache_moves;


std::string cfg_weightsfile;
float cfg_fpu_root_reduction;
float cfg_fpu_reduction;
float cfg_logpuct;
float cfg_puct;
float cfg_logconst;
float cfg_delta_attenuation_ratio;
bool cfg_noise;

int cfg_maintime;
int cfg_byotime;
int cfg_byostones;

int cfg_boardsize;
int cfg_komi;
int cfg_playouts;
float cfg_allow_pass_ratio = 0.0f;

void arg_parser(int argc, char** argv) {
	
	for (int i = 1; i < argc; ++i) {
		std::string cmd(argv[i]);
		if (cmd == "-g" || cmd == "--gtp") {
			gtp::gtp_mode();
		} else if (cmd == "-w" || cmd == "--weight") {
			if ((i+1) <= argc) {
				std::string filename(argv[(i+1)]);
				cfg_weightsfile = filename;
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
	cfg_fpu_root_reduction = 0.25f;
	cfg_fpu_reduction= 0.25f;
	cfg_logpuct = 0.015f;
	cfg_puct = 0.5f;
	cfg_logconst = 1.7f;
	cfg_delta_attenuation_ratio = 0.9f;
	cfg_noise = true;

	cfg_maintime  = 60 * 60 * 100;
	cfg_byotime   = 0;
	cfg_byostones = 0;

	cfg_boardsize = DEFULT_BOARDSIZE;
	cfg_komi = DEFULT_KOMI;
	cfg_playouts = 100;
	cfg_allow_pass_ratio = 0.2f;
}
