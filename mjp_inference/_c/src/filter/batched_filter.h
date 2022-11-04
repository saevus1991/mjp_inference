#pragma once
#include "../../types.h"
// #include "../util/util.h"
#include "../models/mjp.h"
#include "../me/me_inference.h"
#include "../obs_models/obs_model.h"
#include "krylov_filter.h"
#include "krylov_filter_mem.h"

pybind11::tuple batched_filter(np_array_c initial_in, np_array_c rates_in, MEInference* master_equation, ObservationModel* obs_model, np_array_c obs_times_in, np_array_c observations_in, np_array_c obs_param_in, bool get_gradient, int num_workers, std::string backend);

pybind11::tuple batched_filter_list(np_array_c initial_in, np_array_c rates_in, MEInference* master_equation, ObservationModel* obs_model, pybind11::list obs_times_in, pybind11::list observations_in, np_array_c obs_param_in, bool get_gradient, int num_workers, std::string backend);



