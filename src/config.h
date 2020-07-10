#ifndef CONFIG_H_INCLUDE
#define CONFIG_H_INCLUDE

#include <string>

#define MARCRO_BOARDSIZE 19

#define MARCRO_MIN_BOARDSIZE 7
#define MARCRO_MAX_BOARDSIZE 25

static constexpr int BOARD_SIZE = MARCRO_BOARDSIZE;
static constexpr int LETTERBOX_SIZE = BOARD_SIZE + 2;

static constexpr int NUM_VERTICES = LETTERBOX_SIZE * LETTERBOX_SIZE;
static constexpr int NUM_INTERSECTIONS = BOARD_SIZE * BOARD_SIZE;


static constexpr float DEFULT_KOMI = 7.5f;
static constexpr int DEFULT_BOARDSIZE = BOARD_SIZE;

static_assert(BOARD_SIZE % 2 == 1, "");
static_assert(BOARD_SIZE <= MARCRO_MAX_BOARDSIZE, "");
static_assert(BOARD_SIZE >= MARCRO_MIN_BOARDSIZE, "");

//#define DEBUG_CHECK

static const std::string PROGRAM_NAME = "TemplateGo";
static const std::string PROGRAM_VERSION = "Pre-alpha";

#endif
