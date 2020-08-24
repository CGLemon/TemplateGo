#ifndef CFG_H_INCLUDE
#define CFG_H_INCLUDE

#include <cstdio>
#include <cstdlib>
#include <string>

#include "Utils.h"
extern FILE *cfg_logfile_stream;

extern bool cfg_quiet;
extern bool cfg_allowed_suicide;
extern bool cfg_pre_block_superko;
extern bool cfg_gtp_mode;

extern float cfg_softmax_temp;
extern int cfg_cache_moves;
extern float cfg_cache_ratio;

extern std::string cfg_weightsfile;
extern int cfg_uct_threads;
extern float cfg_fpu_root_reduction;
extern float cfg_fpu_reduction;
extern float cfg_logpuct;
extern float cfg_puct;
extern float cfg_logconst;

extern bool cfg_dirichlet_noise;
extern float cfg_delta_attenuation_ratio;

extern int cfg_maintime;
extern int cfg_byotime;
extern int cfg_byostones;

extern bool cfg_auto_quit;

extern std::string cfg_selfplay_config;
extern bool cfg_selfplay_agent;
extern bool cfg_random_move;
extern size_t cfg_random_move_cnt;
extern int cfg_boardsize;
extern float cfg_komi;
extern int cfg_playouts;
extern float cfg_allow_pass_ratio;
extern size_t cfg_random_min_visits;
extern int cfg_batchsize;
extern bool cfg_ponder;

extern float cfg_resign_threshold;

extern size_t cfg_max_game_buffer;
extern bool cfg_collect;

extern std::uint64_t cfg_default_seed;

void init_cfg();
void arg_parser(int argc, char **argv);
void adjust_batchsize(int batchsize);
#endif
