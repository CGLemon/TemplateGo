#include "SearchParameters.h"

SearchParameters::SearchParameters() {
    
    resigned_threshold = option<float>("resigned_threshold");
    allowed_pass_ratio = option<float>("allowed_pass_ratio");
    playouts           = option<int>("playouts");

    random_min_visits = option<int>("random_min_visits");
    dirichlet_noise   = option<bool>("dirichlet_noise");
    ponder            = option<bool>("ponder");
    collect           = option<bool>("collect");

    fpu_root_reduction = option<float>("fpu_root_reduction");
    fpu_reduction      = option<float>("fpu_reduction");
    logconst           = option<float>("logconst");
    logpuct            = option<float>("logpuct");
    puct               = option<float>("puct");
    score_utility_div  = option<float>("score_utility_div");

    komi = option<float>("komi");
}
