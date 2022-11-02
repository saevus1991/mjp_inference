#pragma once
#include <random>
#include <any>

#include "../../types.h"
#include "../util/misc.h"
#include "../models/mjp.h"
#include "../filter/krylov_backward_filter.h"
#include "trajectory.h"
// #include "util.h"

class PosteriorSimulator {

    public:

    // constructor
    PosteriorSimulator(MJP* model_, vec initial_, vec tspan_, std::mt19937* rng_, int max_events_, std::string max_event_handler_);
    PosteriorSimulator(MJP* model_, int num_states, vec rates_, vec tspan_, std::mt19937* rng_, int max_events_, std::string max_event_handler_);
    PosteriorSimulator(MJP* model_, vec initial_, vec rates_in, vec tspan_, std::mt19937* rng_, int max_events_, std::string max_event_handler_);

    // setter
    inline void set_initial(vec& initial_in) {
        initial = initial_in;
    }

    // getter
    inline std::mt19937* get_rng() {
        return(rng);
    }

    // main functions
    template <class S, class T>
    Trajectory simulate(S& t_grid, T& backward_grid);
    pybind11::dict simulate(np_array_c t_grid_in, np_array_c backward_in);
    Trajectory simulate(Eigen::Map<vec>& t_grid, KrylovBackwardFilter& filt);
    template <class S, class T>
    std::tuple<bool, double, int> next_reaction (double time, vec& state, vec& hazard, S& time_grid, T& backward, std::vector<double>& internal_time, std::vector<double>& stats);
    std::tuple<bool, double, int> next_reaction (double time, vec& state, vec& hazard, Eigen::Map<vec>& time_grid, KrylovBackwardFilter& filt, std::vector<double>& internal_time, std::vector<double>& stats);

    // helper functions
    vec get_control(vec& state, vec& backward);
    vec get_control(int ind, std::vector<int>& targets, vec& backward);

    protected:

    MJP* model;
    vec initial;
    vec rates;
    int max_events;
    std::string max_event_handler;
    vec tspan;
    std::mt19937* rng;
    std::uniform_real_distribution<double> U;
    int t_index;

};

class PyPosteriorSimulator : public PosteriorSimulator {
    public:
    PyPosteriorSimulator(MJP* model_, vec initial_, vec rates_, vec tspan_, int seed_, int max_events_, std::string max_event_handler_);
    PyPosteriorSimulator(MJP* model_, vec initial_, vec tspan_, int seed_, int max_events_, std::string max_event_handler_);

    protected:
    std::mt19937 rng_raw;
};