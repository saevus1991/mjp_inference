#pragma once

#include "../../types.h"

namespace rv
{

inline double sample_uniform(std::mt19937* rng) {
    return(std::uniform_real_distribution<double>()(*rng));
}

inline double sample_gauss(std::mt19937* rng) {
    return(std::normal_distribution<double>()(*rng));
}

inline std::map<std::string, RVSampler> get_sampler_map() {
    std::map<std::string, RVSampler> sampler_map;
    sampler_map.insert({"uniform", sample_uniform});
    sampler_map.insert({"gauss", sample_gauss});
    return(sampler_map);
}
    
} // namespace rv
