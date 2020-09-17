#ifndef CONFIG_H_INCLUDE
#define CONFIG_H_INCLUDE
#include <cstdio>
#include <cstdlib>
#include <string>

#include "Utils.h"
#include <string>

#define MARCRO_BOARDSIZE 19

#define MARCRO_MIN_BOARDSIZE 2
#define MARCRO_MAX_BOARDSIZE 25

static constexpr int BOARD_SIZE = MARCRO_BOARDSIZE;
static constexpr int LETTERBOX_SIZE = BOARD_SIZE + 2;

static constexpr int NUM_VERTICES = LETTERBOX_SIZE * LETTERBOX_SIZE;
static constexpr int NUM_INTERSECTIONS = BOARD_SIZE * BOARD_SIZE;

static constexpr float DEFULT_KOMI = 7.5f;
static constexpr int DEFULT_BOARDSIZE = BOARD_SIZE;

static_assert(BOARD_SIZE <= MARCRO_MAX_BOARDSIZE, "");
static_assert(BOARD_SIZE >= MARCRO_MIN_BOARDSIZE, "");

static const std::string PROGRAM_NAME = "TemplateGo";
static const std::string PROGRAM_VERSION = "Alpha";




extern FILE *cfg_logfile_stream;

extern bool cfg_quiet;
extern bool cfg_allowed_suicide;
extern bool cfg_pre_block_superko; // ignore
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

extern bool cfg_selfplay_agent;
extern bool cfg_random_move;
extern int cfg_random_move_cnt;
extern int cfg_random_move_div;
extern int cfg_boardsize;
extern float cfg_komi;
extern int cfg_playouts;
extern float cfg_allow_pass_ratio;
extern size_t cfg_random_min_visits;
extern int cfg_batchsize;
extern int cfg_waittime;
extern bool cfg_ponder;

extern float cfg_resign_threshold;

extern size_t cfg_max_game_buffer;
extern bool cfg_collect;

extern int cfg_lable_komi;
extern int cfg_lable_shift;
extern float cfg_label_buffer;

extern std::uint64_t cfg_default_seed;

void init_cfg();
void arg_parser(int argc, char **argv);
#endif
