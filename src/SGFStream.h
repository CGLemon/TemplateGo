#ifndef SGFSTREAM_H_INCLUDE
#define SGFSTREAM_H_INCLUDE
#include "GameState.h"
#include "Board.h"

#include <string>
#include <iostream>

class SGFstream {
public:
    static void save_sgf(std::string filename, GameState &state, bool append = false);

private:
    static void sgf_stream(std::ostream &out, GameState &state);

};

#endif
