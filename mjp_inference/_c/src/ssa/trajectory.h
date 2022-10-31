#pragma once

#include "../../types.h"
#include "../util/conversion.h"
#include "../models/mjp.h"

struct Trajectory {

    vec initial;
    vec tspan;
    std::vector<double> time;
    std::vector<int> events;
    mat_rm states;

    // constructors
    Trajectory() = default;
    Trajectory(pybind11::dict trajectory);

    // static functions
    static mat_rm construct_trajectory(MJP* model, const std::vector<int>& events, vec& initial);

    // helper functions
    pybind11::dict to_dict();
    inline void clear() {
        *this = Trajectory();
    }

};
