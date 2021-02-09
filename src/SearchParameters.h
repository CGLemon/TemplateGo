#ifndef SEARCHPARAMETERS_H_INCLUDE
#define SEARCHPARAMETERS_H_INCLUDE

#include "config.h"

class SearchParameters {
public:
    SearchParameters();
    float resigned_threshold;
    float allowed_pass_ratio;
    int playouts;
    int random_min_visits;

    bool dirichlet_noise;
    bool ponder;
    bool collect;

    double fpu_root_reduction;
    double fpu_reduction;
    double logconst;
    double logpuct;
    double puct;
    double score_utility_div;

    float komi;
};

#endif
