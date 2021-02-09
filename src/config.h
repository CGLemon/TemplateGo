#ifndef CONFIG_H_INCLUDE
#define CONFIG_H_INCLUDE

#include <string>
#include <unordered_map>


#define MARCO_MAXIMAL_GTP_BOARD_SIZE (25)
#define MARCO_MINIMAL_GTP_BOARD_SIZE (2)

// The maximal of board size.
#define MARCO_BOARD_SIZE (9)

// Avoid crazy komi.
#define MARCO_MAXIMAL_KOMI (150.f)
#define MARCO_MINIMAL_KOMI (-150.f)
#define MARCO_KOMI (7.0f)

static_assert(MARCO_MAXIMAL_GTP_BOARD_SIZE >= MARCO_BOARD_SIZE &&
                  MARCO_BOARD_SIZE >= MARCO_MINIMAL_GTP_BOARD_SIZE, "Not support for this board size!\n");

static constexpr int BOARD_SIZE = MARCO_BOARD_SIZE;

static constexpr int LETTERBOX_SIZE = BOARD_SIZE + 2;

static constexpr int NUM_INTERSECTIONS = BOARD_SIZE * BOARD_SIZE;

static constexpr int NUM_VERTICES = LETTERBOX_SIZE * LETTERBOX_SIZE;

static constexpr auto DEFAULT_KOMI = MARCO_KOMI;

static constexpr auto DEFAULT_BOARDSIZE = BOARD_SIZE;

const std::string PROGRAM = "TemplateGo";

const std::string VERSION = "Alpha"; 

extern bool cfg_quiet;

template<typename T>
T option(std::string name);

template<typename T>
bool set_option(std::string name, T val);

void init_basic_parameters();

class ArgsParser {
public:
    ArgsParser() = delete;

    ArgsParser(int argc, char** argv);

    void dump() const;

private:
    void help() const;
};




#endif
