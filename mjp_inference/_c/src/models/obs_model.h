#pragma once

#include "../../types.h"
#include "mjp.h"
#include "../util/sample_rv.h"


class ObservationModel {

    public:
    // constructor
    ObservationModel(MJP* transition_model_, const std::vector<std::string>& rv_list_, pybind11::tuple transformation_callable_, pybind11::tuple sample_callable_, pybind11::tuple llh_callable_, unsigned transform_dim_, unsigned obs_dim_);

    // helpers
    std::vector<RVSampler> build_rv_map();

    // main functions
    inline vec sample_rv(std::mt19937* rng) {
        vec rv(rv_map.size());
        for (unsigned i = 0; i < rv_map.size(); i++) {
            rv[i] = rv_map[i](rng);
        }
        return(rv);
    }
    inline vec transform(double time, vec& state, vec& param) {
        vec transformed(transform_dim);
        transformation_fun(time, state.data(), param.data(), transformed.data());
        return(transformed);
    }
    inline double llh(double time, vec& state, vec& param) {
        // transform state
        vec transformed = transform(time, state, param);
        // evaluate llh
        double llh = llh_fun(time, transformed.data(), param.data());
        return(llh);
    }
    inline vec sample(double time, vec& state, vec&param, std::mt19937* rng) {
        // sample required rvs
        vec rv = sample_rv(rng);
        // transform state
        vec transformed = transform(time, state, param);
        // generate the sample
        vec obs(obs_dim);
        sample_fun(time, transformed.data(), param.data(), rv.data(), obs.data());
        return(obs);
    }
    inline vec sample_np(double time, vec& state, vec&param, unsigned seed) {
        std::mt19937 rng(seed);
        return(sample(time, state, param, &rng));
    }


    private:
    MJP* transition_model;
    std::vector<std::string> rv_list;
    std::vector<RVSampler> rv_map;
    pybind11::tuple transformation_callable; 
    pybind11::capsule transformation_capsule;
    Transformation transformation_fun; 
    pybind11::tuple sample_callable;
    pybind11::capsule sample_capsule;
    Sampler sample_fun;
    pybind11::tuple llh_callable;
    pybind11::capsule llh_capsule;
    Llh llh_fun;
    unsigned transform_dim;
    unsigned obs_dim;
    mat_rm state_map;

};
