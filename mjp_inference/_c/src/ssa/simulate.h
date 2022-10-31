#pragma once
#include "simulator.h"
#include "posterior_simulator.h"
#include "../obs_models/obs_model.h"
// #include "../filter/krylov_backward_filter.h"
#include "../me/me_inference.h"


np_array simulate(np_array_c initial_in, MJP* transition_model, ObservationModel* obs_model, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler);

pybind11::dict simulate_full(np_array_c initial_in, MJP* transition_model, np_array_c tspan_, int seed, int max_events, std::string max_event_handler);

np_array simulate_batched(np_array_c initial_dist_in, np_array_c rates_in, MJP* transition_model, ObservationModel* obs_model, np_array_c obs_times_in,np_array_c obs_param_in, np_array_c tspan_in, int seed, int num_samples, int num_workers, int max_events, std::string max_event_handler);

// #TODO: add batched version of simuate_full

// np_array simulate(np_array_c initial_in, TransitionModel& transition_model, np_array_c t_eval_in, int seed, int max_events, std::string max_event_handler);

pybind11::object simulate_posterior(np_array_c initial_dist_in, MJP* transition_model, ObservationModel* obs_model, np_array_c t_obs_in, np_array_c observations_in, np_array_c tspan_in, np_array_c t_grid_in, int seed, int num_samples, int max_events, std::string max_event_handler);


pybind11::list simulate_posterior_batched(np_array_c initial_dist_in, np_array_c rates_in, MJP* transition_model, ObservationModel* obs_model, np_array_c obs_times_in, np_array_c observations_in, np_array_c obs_param_in, np_array_c tspan_in, np_array_c t_grid_in, int seed, int num_workers, int max_events, std::string max_event_handler);
