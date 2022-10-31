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
    PosteriorSimulator(MJP* model_in, const vec& initial_in, const vec& tspan_in, int seed_in, int max_events_in, const std::string& max_event_handler_in);
    PosteriorSimulator(MJP* model_in, int num_states, const vec& rates_in, const vec& tspan_in, int seed_in, int max_events_in, const std::string& max_event_handler_in);
    // Simulator(TransitionModel& model_in, vec tspan_in);

    // setter
    inline void set_initial(vec& initial_in) {
        initial = initial_in;
    }

    // getter
    inline std::mt19937& get_rng() {
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

    private:

    MJP* model;
    vec initial;
    vec rates;
    int seed;
    int max_events;
    std::string max_event_handler;
    vec tspan;
    std::mt19937 rng;
    std::uniform_real_distribution<double> U;
    int t_index;

};