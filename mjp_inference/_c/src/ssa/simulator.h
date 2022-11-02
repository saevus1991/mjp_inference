#pragma once

#include <random>

#include "../../types.h"
#include "../models/mjp.h"
#include "../util/math.h"
#include "trajectory.h"

class Simulator {

    public:

    // constructor
    Simulator(MJP* model_, vec initial_, vec tspan_, int seed_, int max_events_, std::string max_event_handler_);
    Simulator(MJP* model_, int num_states, vec rates_, vec tspan_, int seed_, int max_events_, std::string max_event_handler_);
    Simulator(MJP* model_, vec initial_, vec rates_, vec tspan_, int seed_, int max_events_, std::string max_event_handler_);

    // getters
    inline std::mt19937& get_rng() {
        return(rng);
    }
    // setters
    inline void set_initial(vec& initial_) {
        initial = initial_;
    }

    // main functions
    Trajectory simulate();
    pybind11::dict simulate_py();
    np_array simulate(np_array_c t_eval);
    mat_rm simulate(const Eigen::Map<vec>& t_eval);
    std::tuple<bool, double, int> next_reaction(double time, vec& hazard);


    private:
    MJP* model;
    vec initial;
    vec rates;
    int seed;
    int max_events;
    std::string max_event_handler;
    vec tspan;
    std::mt19937 rng; // #TODO: make simulate store pointer to rng
    std::uniform_real_distribution<double> U;

};