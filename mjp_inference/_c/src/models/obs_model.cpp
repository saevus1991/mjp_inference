#include "obs_model.h"

// constructor

ObservationModel::ObservationModel(MJP* transition_model_, const std::vector<std::string>& rv_list_, pybind11::tuple transformation_callable_, pybind11::tuple sample_callable_, pybind11::tuple llh_callable_, unsigned transform_dim_, unsigned obs_dim_) :
    transition_model(transition_model_),
    rv_list(rv_list_),
    rv_map(build_rv_map()),
    transformation_callable(transformation_callable_),
    transformation_capsule(transformation_callable[0]),
    transformation_fun(reinterpret_cast<void (*)(double, double*, double*, double*)>(transformation_capsule.get_pointer())),
    sample_callable(sample_callable_),
    sample_capsule(sample_callable[0]),
    sample_fun(reinterpret_cast<void (*)(double, double*, double*, double*, double*)>(sample_capsule.get_pointer())),
    llh_callable(llh_callable_),
    llh_capsule(llh_callable[0]),
    llh_fun(reinterpret_cast<double (*)(double, double*, double*)>(llh_capsule.get_pointer())),
    transform_dim(transform_dim_),
    obs_dim(obs_dim_),
    state_map(transition_model->build_state_map())
    {}

// helpers

std::vector<RVSampler> ObservationModel::build_rv_map() {
    std::map<std::string, RVSampler> sampler_map = rv::get_sampler_map();
    std::vector<RVSampler> rv_map;
    for (unsigned i = 0; i < rv_list.size(); i++) {
        std::string rv_type = rv_list[i];
        rv_map.push_back(sampler_map[rv_type]);
    }
    return(rv_map);
}

// vec ObservationModel::sample_rv(std::mt19937* rng) {
//     vec rv(rv_list.size());
// }