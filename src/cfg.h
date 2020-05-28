#ifndef CFG_H_INCLUDE
#define CFG_H_INCLUDE

#include <cstdio>
#include <cstdlib>
#include <string>
extern FILE * cfg_logfile_stream;

extern bool cfg_quiet;
extern bool cfg_allowed_suicide;
extern bool cfg_pre_block_superko;
extern bool cfg_gtp_mode;

extern float cfg_softmax_temp;
extern int cfg_cache_moves;

extern std::string cfg_weightsfile;
extern float cfg_fpu_root_reduction;
extern float cfg_fpu_reduction;
extern float cfg_logpuct;
extern float cfg_puct;
extern float cfg_logconst;

extern bool cfg_noise;
extern float cfg_delta_attenuation_ratio;


extern int cfg_maintime;
extern int cfg_byotime;
extern int cfg_byostones;


extern int cfg_boardsize;
extern int cfg_komi;
extern int cfg_playouts;
extern float cfg_allow_pass_ratio;

void init_cfg();
void arg_parser(int argc, char** argv);



#endif
