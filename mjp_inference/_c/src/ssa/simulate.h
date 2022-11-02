#pragma once
#include "simulator.h"
#include "posterior_simulator.h"
#include "../obs_models/obs_model.h"
// #include "../filter/krylov_backward_filter.h"
#include "../me/me_inference.h"
#include "../util/conversion.h"

// simulate on discrete grid

mat_rm simulate(MJP* transition_model, ObservationModel* obs_model, const Eigen::Map<vec>& initial, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& obs_param, const Eigen::Map<vec>& t_eval, int seed, int max_events, std::string max_event_handler);

np_array simulate(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_in, np_array_c rates_in,  np_array_c obs_param_in, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler);

np_array simulate(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_in, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler);

np_array simulate(MJP* transition_model, np_array_c initial_in, np_array_c rates_in, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler);

np_array simulate(MJP* transition_model, np_array_c initial_in, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler);

// simulate full trajectory 

inline Trajectory simulate_full(MJP* transition_model, const Eigen::Map<vec>& initial, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& tspan, int seed, int max_events, std::string max_event_handler) {
    std::mt19937 rng(seed);
    Simulator simulator(transition_model, initial, rates, tspan, &rng, max_events, max_event_handler);
    return(simulator.simulate());
}

pybind11::dict simulate_full(MJP* transition_model, np_array_c initial_, np_array_c rates_, np_array_c tspan_, int seed, int max_events, std::string max_event_handler);

pybind11::dict simulate_full(MJP* transition_model, np_array_c initial_, np_array_c tspan_, int seed, int max_events, std::string max_event_handler);

// batched simulator

np_array simulate_batched(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c rates_in, np_array_c obs_param_in, np_array_c obs_times_in, int seed, int num_samples, int num_workers, int max_events, std::string max_event_handler);

// #TODO: add batched version of simuate_full

// np_array simulate(np_array_c initial_in, TransitionModel& transition_model, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler);

pybind11::object simulate_posterior(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c t_obs_in, np_array_c observations_in, np_array_c tspan_in, np_array_c t_grid_in, int seed, int num_samples, int max_events, std::string max_event_handler);


pybind11::list simulate_posterior_batched(MJP* transition_model, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c rates_in, np_array_c obs_times_in, np_array_c observations_in, np_array_c obs_param_in, np_array_c tspan_in, np_array_c t_grid_in, int seed, int num_workers, int max_events, std::string max_event_handler);
