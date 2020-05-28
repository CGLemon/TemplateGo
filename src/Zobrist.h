
#ifndef ZOBRIST_H_INCLUDE
#define ZOBRIST_H_INCLUDE

#include <random>
#include <array>
#include <vector>

#include "config.h"


class Zobrist {
public:
	static constexpr std::uint64_t zobrist_seed 		= 0xabcdabcd12345678;
    static constexpr std::uint64_t zobrist_empty 		= 0x1234567887654321;
    static constexpr std::uint64_t zobrist_blacktomove 	= 0xabcdabcdabcdabcd;

	static std::array<std::array<std::uint64_t, NUM_VERTICES>, 4>  	  zobrist;
	static std::array<std::uint64_t, NUM_VERTICES>                    zobrist_ko;
	static std::array<std::array<std::uint64_t, NUM_VERTICES * 2>, 2> zobrist_pris;
	static std::array<std::uint64_t, 5>                               zobrist_pass;

 
    static void init_zobrist();
};


#endif

