#pragma once
#include "simulator.h"
#include "posterior_simulator.h"
#include "../obs_models/obs_model.h"
// #include "../filter/krylov_backward_filter.h"
#include "../me/me_inference.h"
#include "../util/conversion.h"

// simulate on discrete grid

mat_rm simulate(MJP* transition_model, ObservationModel* obs_model, const Eigen::Map<vec>& initial, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& obs_param, const Eigen::Map<vec>& t_eval, std::mt19937* rng, int max_events, std::string max_event_handler);

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

Trajectory simulate_posterior(MJP* transition_model, const Eigen::Map<vec>& initial, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& tspan, const Eigen::Map<vec>& t_grid, const Eigen::Map<mat_rm>& backward_grid, std::mt19937* rng, int max_events, std::string max_event_handler);

Trajectory simulate_posterior(MEInference* master_equation, ObservationModel* obs_model, const Eigen::Map<vec>& initial_dist, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& obs_param, const Eigen::Map<vec>& tspan, const Eigen::Map<vec>& obs_times, const Eigen::Map<mat_rm>& observations, const Eigen::Map<vec>& t_grid, std::mt19937* rng, int max_events, std::string max_event_handler);

Trajectory simulate_posterior(MEInference* master_equation, ObservationModel* obs_model, const Eigen::Map<vec>& initial_dist, const Eigen::Map<vec>& rates, const Eigen::Map<vec>& obs_param, const Eigen::Map<vec>& tspan, const Eigen::Map<vec>& obs_times, const Eigen::Map<mat_rm>& observations, const Eigen::Map<vec>& t_grid, int seed, int max_events, std::string max_event_handler);


pybind11::dict simulate_posterior(MEInference* master_equation, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c rates_in, np_array_c obs_param_in, np_array_c tspan_in, np_array_c obs_times_in, np_array_c observations_in, np_array_c t_grid_in, int seed,  int max_events, std::string max_event_handler);


pybind11::list simulate_posterior_batched(MEInference* master_equation, ObservationModel* obs_model, np_array_c initial_dist_in, np_array_c rates_in, np_array_c obs_param_in, np_array_c tspan_in, np_array_c obs_times_in, np_array_c observations_in, np_array_c t_grid_in, int seed, int num_samples, int num_workers, int max_events, std::string max_event_handler);
